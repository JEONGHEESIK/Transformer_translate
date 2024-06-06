import torch
from transformers import Transformer
from util import CustomTokenizer

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
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

def translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device):
    tokens = src_tokenizer.encode(sentence, max_length)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    trg_indices = [trg_tokenizer.vocab["<sos>"]]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        best_guess = output.argmax(2)[:, -1].item()
        trg_indices.append(best_guess)
        if best_guess == trg_tokenizer.vocab["<eos>"]:
            break

    translated_tokens = [trg_tokenizer.decode([i]) for i in trg_indices]
    return " ".join(translated_tokens)

# Example usage
sentence = "나는 고양이를 좋아합니다"
translation = translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device)
print(f"Translated: {translation}")