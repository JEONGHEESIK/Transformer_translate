import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import BertTokenizer
from model import Transformer
from util import ExcelTranslationDataset

# Pre-trained Tokenizers from Hugging Face
src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
trg_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Vocabulary sizes
src_vocab_size = src_tokenizer.vocab_size
trg_vocab_size = trg_tokenizer.vocab_size
src_pad_idx = src_tokenizer.pad_token_id
trg_pad_idx = trg_tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 10

# Model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# Dataset and DataLoader
file_path = "C:/Users/JEONGHEESIK/Desktop/Coding/transformer/corpus_eng.xlsx"
dataset = ExcelTranslationDataset(file_path, src_tokenizer, trg_tokenizer, max_length, max_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Accuracy calculation
def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).float()
    acc = correct.sum() / len(correct)
    return acc

# Training Loop
for epoch in range(10):
    model.train()
    train_loss = 0
    train_acc = 0
    for src, trg in train_loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.reshape(-1, output.shape[2])
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        acc = calculate_accuracy(output, trg)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for src, trg in val_loader:
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg[:, :-1])
            output = output.reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            acc = calculate_accuracy(output, trg)

            val_loss += loss.item()
            val_acc += acc.item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the model
torch.save(model.state_dict(), "transformer_model.pth")

# Example usage for translation
def translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device):
    tokens = src_tokenizer.encode(sentence, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
    trg_indices = [trg_tokenizer.cls_token_id]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tokens, trg_tensor)
        best_guess = output.argmax(2)[:, -1].item()
        trg_indices.append(best_guess)
        if best_guess == trg_tokenizer.sep_token_id:
            break

    translated_tokens = trg_tokenizer.convert_ids_to_tokens(trg_indices)
    return " ".join(translated_tokens)

# Load the trained model for translation
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

sentence = "나는 고기를 좋아합니다"
translation = translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device)
print(f"Translated: {translation}")
