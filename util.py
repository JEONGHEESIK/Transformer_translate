import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def tokenize(self, sentence):
        return sentence.split()
    
    def encode(self, sentence, max_length):
        tokens = self.tokenize(sentence)
        token_ids = [self.vocab.get(token, 0) for token in tokens]  # 미등록 단어는 0으로
        if len(token_ids) < max_length:
            token_ids += [0] * (max_length - len(token_ids))  # 패딩
        elif len(token_ids) > max_length:
            token_ids = token_ids[:max_length]  # 잘라내기
        return token_ids

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
        src = row.iloc[0]  # 첫 번째 열이 소스 문장
        trg = row.iloc[1]  # 두 번째 열이 타겟 문장
        
        # src와 trg가 문자열인지 확인
        if not isinstance(src, str):
            src = str(src)
        if not isinstance(trg, str):
            trg = str(trg)
        
        src_token_ids = self.src_tokenizer.encode(src, self.src_max_length)
        trg_token_ids = self.trg_tokenizer.encode(trg, self.trg_max_length)

        return torch.tensor(src_token_ids), torch.tensor(trg_token_ids)