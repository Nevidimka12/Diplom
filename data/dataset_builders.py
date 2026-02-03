import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_embedding_dict(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)
    keys = data["keys"]
    emb_dict = {}
    for i, k in enumerate(keys):
        emb = data[f"v_{i}"]  # (T_i, D)
        emb_dict[str(k)] = emb
    return emb_dict



def get_embedding(emb_dict: Dict[str, np.ndarray], key: str) -> torch.Tensor:
    emb = emb_dict[key]  # (1, T, H) или (T, H)
    emb = np.array(emb)
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]  # (T, H)
    elif emb.ndim != 2:
        raise ValueError(f"Ожидается (1,T,H) или (T,H), а получено {emb.shape} для key={key}")
    return torch.from_numpy(emb).float()  # (T, H)



# CMU-MOSEI:

class CMUMOSEIEmotionDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        emotion_columns: Optional[List[str]] = None,
        dataset_name: str = "cmu_mosei",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)

        self.key_column = "video_name"
        if emotion_columns is None:
            # стандартный порядок
            emotion_columns = [
                "Neutral", "Anger", "Disgust", "Fear",
                "Happiness", "Sadness", "Surprise"
            ]
        self.emotion_columns = emotion_columns

        valid_rows = []
        for _, row in self.df.iterrows():
            key = str(row[self.key_column])
            if key in self.emb_dict:
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "CMUMOSEIEmotionDataset: ни одной строки не совпало по ключам с эмбеддингами."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = str(row[self.key_column])

        x = get_embedding(self.emb_dict, key)  # (T, H)
        length = x.shape[0]

        vals = pd.to_numeric(row[self.emotion_columns], errors="coerce").astype("float32").to_numpy()
        emo_vec = torch.from_numpy(vals)  # (7,)

        return {
            "x": x,
            "length": length,
            "emotion_mosei": emo_vec,
            "emotion_resd": None,
            "personality": None,
            "ah": None,
            "dataset_name": self.dataset_name,
        }



# RESD:

class RESDEmotionDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        dataset_name: str = "resd",
        label_column: str = "emotion",
        label_mapping: Dict[str, int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)
        self.key_column = "name"
        self.label_column = label_column

        if label_mapping is None:
            raise ValueError("RESDEmotionDataset: label_mapping должен быть передан снаружи.")
        self.label_mapping = label_mapping

        valid_rows = []
        for _, row in self.df.iterrows():
            base_name = str(row[self.key_column])
            key = base_name
            if key in self.emb_dict:
                row = row.copy()
                row["_emb_key"] = key
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "RESDEmotionDataset: ни одной строки не совпало по ключам с эмбеддингами."
            )

        unique_labels = sorted(self.df[self.label_column].unique())
        missing = [str(lbl) for lbl in unique_labels if str(lbl) not in self.label_mapping]
        if missing:
            raise ValueError(f"В label_mapping нет классов из csv: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = row["_emb_key"]

        x = get_embedding(self.emb_dict, key)  # (T, H)
        length = x.shape[0]

        raw_label = str(row[self.label_column])
        label = self.label_mapping[raw_label]

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": label,
            "personality": None,
            "ah": None,
            "dataset_name": self.dataset_name,
        }


# FirstImpressionsV2:

class FIV2PersonalityDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        subset: str,
        dataset_name: str = "fiv2",
        personality_columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)

        subset = subset.lower()
        if subset not in {"train", "test"}:
            raise ValueError("subset должен быть 'train' или 'test'")
        self.df = self.df[self.df["Subset"].str.lower() == subset].reset_index(drop=True)

        self.key_column = "NAME_VIDEO"
        if personality_columns is None:
            personality_columns = [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "non-neuroticism",
            ]
        self.personality_columns = personality_columns

        valid_rows = []
        for _, row in self.df.iterrows():
            raw_name = str(row[self.key_column])
            file_id = os.path.splitext(raw_name)[0]
            if file_id in self.emb_dict:
                row = row.copy()
                row["_emb_key"] = file_id
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                f"FIV2PersonalityDataset({subset}): ни одной строки не совпало с эмбеддингами."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = row["_emb_key"]

        x = get_embedding(self.emb_dict, key)  # (T, H)
        length = x.shape[0]

        vals = pd.to_numeric(
            row[self.personality_columns],
            errors="coerce"
        ).astype("float32").to_numpy()
        pers = torch.from_numpy(vals)  # (5,)

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": None,
            "personality": pers,
            "ah": None,
            "dataset_name": self.dataset_name,
        }


# BAH:

def load_bah_split_paths(split_txt_path: str) -> List[str]:
    paths = []
    with open(split_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",", 1)[0].strip()
            paths.append(first)
    return paths


class BAHAbsencePresenceDataset(Dataset):
    def __init__(
        self,
        embeddings_npy_path: str,
        labels_csv_path: str,
        split_txt_path: str,
        dataset_name: str = "bah",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.emb_dict = load_embedding_dict(embeddings_npy_path)

        self.df = pd.read_csv(labels_csv_path)
        self.key_column = "video-path"
        self.label_column = "label"

        split_paths = set(load_bah_split_paths(split_txt_path))

        valid_rows = []
        for _, row in self.df.iterrows():
            key = str(row[self.key_column])
            if key in split_paths and key in self.emb_dict:
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "BAHAbsencePresenceDataset: ни одной строки не совпало по сплиту и эмбеддингам."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = str(row[self.key_column])

        x = get_embedding(self.emb_dict, key)  # (T, H)
        length = x.shape[0]

        label = int(row[self.label_column])

        return {
            "x": x,
            "length": length,
            "emotion_mosei": None,
            "emotion_resd": None,
            "personality": None,
            "ah": label,
            "dataset_name": self.dataset_name,
        }
