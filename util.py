import pandas as pd
import torch
from torch.utils.data import Dataset

class ExcelTranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, trg_tokenizer, src_max_length, trg_max_length):
        self.data = pd.read_excel(file_path)
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_max_length = src_max_length
        self.trg_max_length = trg_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src = row[0]
        trg = row[1]

        src_token_ids = self.src_tokenizer.encode(src, add_special_tokens=True, max_length=self.src_max_length, truncation=True, padding='max_length')
        trg_token_ids = self.trg_tokenizer.encode(trg, add_special_tokens=True, max_length=self.trg_max_length, truncation=True, padding='max_length')

        return torch.tensor(src_token_ids), torch.tensor(trg_token_ids)
