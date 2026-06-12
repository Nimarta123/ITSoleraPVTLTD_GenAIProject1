# ITSoleraPVTLTD_GenAIProject1
#Multimodal Data Preprocessing Pipeline
#This module creates the unified dataset by standardizing medical images and extracting clinical entities using a biomedical NLP parser.
import os
import ast
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import spacy

# Ensure biomedical NLP model is available
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    import os
    os.system("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
    nlp = spacy.load("en_core_sci_sm")

class BaseMedicalDataset(Dataset):
    """
    Core template handling global image normalization and clinical text entity extraction
    """
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _normalize_and_resize_image(self, img_path):
        # Read image supporting varying bits-per-pixel depths
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Medical image missing at target location: {img_path}")

        # Standardize grayscale (Chest X-Rays) vs RGB (Pathology slides) to a standard 3-Channel Space
        if len(img.shape) == 2:
            # Apply CLAHE to resolve scanner variance contrast disparities
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if img.shape[2] == 4: # Drop alpha transparency layers if present
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale image to matching pixel area footprint
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return self.transform(img_resized)

    def _extract_entities(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return []
        doc = nlp(text.lower())
        return list(set([ent.text for ent in doc.ents]))


class MIMICCXRAdapter(BaseMedicalDataset):
    """
    Adapter targeting the MIMIC-CXR database layout on PhysioNet
    Requires matching the reports metadata tracking sheet to section files
    """
    def __init__(self, csv_metadata_path, reports_dir, images_dir, target_size=(512, 512)):
        super().__init__(target_size)
        self.df = pd.read_csv(csv_metadata_path)
        self.reports_dir = reports_dir
        self.images_dir = images_dir
        
        # Drop entries missing a valid paired report tracking pointer
        self.df = self.df.dropna(subset=['study_id'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Reconstruct path using MIMIC standard storage tree layout: p[patient_id_prefix]/p[patient_id]/s[study_id]
        pid = str(row['subject_id'])
        sid = str(row['study_id'])
        dicom_id = str(row['dicom_id'])
        
        img_path = os.path.join(self.images_dir, f"p{pid[:2]}", f"p{pid}", f"s{sid}", f"{dicom_id}.png")
        report_path = os.path.join(self.reports_dir, f"p{pid[:2]}", f"p{pid}", f"s{sid}.txt")

        # Load raw radiology report text file
        raw_report = ""
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                raw_report = f.read()

        image_tensor = self._normalize_and_resize_image(img_path)
        entities = self._extract_entities(raw_report)

        return {
            "dataset_origin": "MIMIC-CXR",
            "patient_id": f"MIMIC_{pid}",
            "image": image_tensor,
            "raw_report": raw_report,
            "entities": entities,
            "bbox": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32) # Base dataset contains no local annotations
        }


class PadChestAdapter(BaseMedicalDataset):
    """
    Adapter targeting the BIMCV PadChest tracking catalog matrix layout
    Extracts manual spatial geometric annotations when present
    """
    def __init__(self, csv_path, images_dir, target_size=(512, 512)):
        super().__init__(target_size)
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, str(row['ImageID']))
        
        raw_report = str(row.get('Report', ''))
        image_tensor = self._normalize_and_resize_image(img_path)
        entities = self._extract_entities(raw_report)

        # Parse normalized coordinate targets out from label strings if available
        bbox = [0.0, 0.0, 0.0, 0.0]
        if 'LabelsLocalizer' in row and pd.notna(row['LabelsLocalizer']):
            try:
                # Attempt to extract bounding coordinate list formats inside strings
                coords = ast.literal_eval(row['LabelsLocalizer'])
                if isinstance(coords, list) and len(coords) >= 4:
                    bbox = [float(c) for c in coords[:4]]
            except Exception:
                pass

        return {
            "dataset_origin": "PadChest",
            "patient_id": f"PAD_{row['PatientID']}",
            "image": image_tensor,
            "raw_report": raw_report,
            "entities": entities,
            "bbox": torch.tensor(bbox, dtype=torch.float32)
        }


class OpenPathAdapter(BaseMedicalDataset):
    """
    Adapter targeting the OpenPath Digital Pathology Image-Text pipeline layout
    Handles multi-channel microscopy configurations matched to diagnostic text files
    """
    def __init__(self, metadata_json_or_csv, images_dir, target_size=(512, 512)):
        super().__init__(target_size)
        self.images_dir = images_dir
        
        if metadata_json_or_csv.endswith('.json'):
            with open(metadata_json_or_csv, 'r') as f:
                self.data_list = json.load(f)
            self.is_json = True
        else:
            self.df = pd.read_csv(metadata_json_or_csv)
            self.is_json = False

    def __len__(self):
        return len(self.data_list) if self.is_json else len(self.df)

    def __getitem__(self, idx):
        if self.is_json:
            item = self.data_list[idx]
            filename = item['image_path']
            raw_report = item['caption']
            patient_id = item.get('sample_id', str(idx))
        else:
            row = self.df.iloc[idx]
            filename = row['image_path']
            raw_report = row['caption']
            patient_id = row.get('sample_id', str(idx))

        img_path = os.path.join(self.images_dir, filename)
        image_tensor = self._normalize_and_resize_image(img_path)
        entities = self._extract_entities(raw_report)

        return {
            "dataset_origin": "OpenPath",
            "patient_id": f"PATH_{patient_id}",
            "image": image_tensor,
            "raw_report": raw_report,
            "entities": entities,
            "bbox": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        }
