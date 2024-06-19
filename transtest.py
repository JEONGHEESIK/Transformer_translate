import torch
from transformers import BertTokenizer
from model import Transformer

# Load BERT tokenizer
src_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
trg_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Parameters
src_vocab_size = src_tokenizer.vocab_size
trg_vocab_size = trg_tokenizer.vocab_size
src_pad_idx = src_tokenizer.pad_token_id
trg_pad_idx = trg_tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 5

# Model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

def translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device):
    tokens = src_tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length')
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    trg_indices = [trg_tokenizer.cls_token_id]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        best_guess = output.argmax(2)[:, -1].item()
        trg_indices.append(best_guess)
        if best_guess == trg_tokenizer.sep_token_id:
            break

    translated_tokens = trg_tokenizer.convert_ids_to_tokens(trg_indices)
    return " ".join(translated_tokens).replace(" ##", "")

# Example usage
sentence = "I like eating meat"
translation = translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, max_length, device)
print(f"Translated: {translation}")
