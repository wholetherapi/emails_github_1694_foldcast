import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class StructureDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(f"{root}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        record = np.load(self.files[idx])
        coords = torch.tensor(record["coords"], dtype=torch.float32)
        features = torch.tensor(record["features"], dtype=torch.float32)
        edge_index = torch.tensor(record["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(record["edge_attr"], dtype=torch.float32)
        return {
            "coords": coords,
            "features": features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }


def collate(batch):
    return {
        "coords": torch.stack([b["coords"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
    }
