# 트랜스포머 번역 모델
## 개요
- 병렬 언어 데이터셋을 이용한 Transforemr 번역 모델 설계

#### 만든 목적
- Transformer 모델 설계를 통한 구조 이해
- 병렬 언어 데이터셋, 토크나이저 데이터 처리 과정 이해

#### 일정
- 24.03. ~ 24.06.
- 참여도 : 100% (개인 프로젝트)

## 사용 기술 및 개발 환경
- O/S : macOS 14.5 (Sonoma), Window 10
- DB : MySQL
- Framework : PyTorch
- Language : Python
- Tool : GitHub, SQL Developer, VSCode

## 내용
#### 구현 기능

##### Multi Head Attention 클래스
- queries, keys, values 를 사용해 Attention 계산.

```
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        # 생략
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 생략 
```

##### TransformerBlock 클래스
- Transformer의 하나의 블록을 구현, Multi-Head Attention, Layer Normalization, Residual Connection, 그리고 Feed Forward Network를 포함.

```
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), # fully connected layer
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size), # fully connected layer
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
```

##### Encoder 클래스
- 입력 문장을 임베딩하고, 포지션 임베딩을 더한 후 여러 개의 TransformerBlock을 통과시켜 인코딩하는 클래스.

```
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
```

##### DecoderBlock 클래스
- Self-Attention, Encoder-Decoder Attention, Feed Forward Network를 포함

```
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
```

##### Decoder 클래스
- 입력 문장을 임베딩하고, 포지션 임베딩을 더한 후 여러 개의 DecoderBlock을 통과시켜 디코딩. fully connected layer를 통과시켜 단어 예측을 수행.

```
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size) # fully connected layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out
```

##### Transformer 클래스
- 전체 Transformer 모델을 정의. 인코더와 디코더 포함, 입력과 출력 마스크를 생성. 입력 문장을 인코딩한 후, 디코더를 통해 번역을 수행

```
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
```


#### output
- Current Progress

##### accuracy problem
###### the speed of learning problem
###### overfitting problem
└ hyper-parameter changing, dataset reduction etc.. processing
