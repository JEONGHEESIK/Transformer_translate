import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import Transformer
from util import CustomTokenizer, ExcelTranslationDataset
import torch.cuda.amp as amp
import torch.profiler

# Vocabulary and Tokenizers
vocab = {"<pad>": 0, "나는": 1, "고양이를": 2, "좋아합니다": 3, "<sos>": 4, "<eos>": 5}
src_tokenizer = CustomTokenizer(vocab)
trg_tokenizer = CustomTokenizer(vocab)

# Parameters
src_vocab_size = len(vocab)
trg_vocab_size = len(vocab)
src_pad_idx = vocab["<pad>"]
trg_pad_idx = vocab["<pad>"]
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
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

# Accuracy calculation
def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).float()
    acc = correct.sum() / len(correct)
    return acc

# Mixed Precision Training
scaler = amp.GradScaler()

# Profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Training Loop
    for epoch in range(10):
        model.train()
        train_loss = 0
        train_acc = 0
        for src, trg in train_loader:
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                output = model(src, trg[:, :-1])
                output = output.reshape(-1, output.shape[2])
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)
                acc = calculate_accuracy(output, trg)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

                with amp.autocast():
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
        
        prof.step()

# Save the model
torch.save(model.state_dict(), "transformer_model.pth")