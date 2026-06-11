from sympy import sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6
    
    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000

    src_sequence_length: int = 512
    tgt_sequence_length: int = 512
    learn_pos_embedd: bool = False

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedd_dim, requires_grad=False):
        super().__init__()

        self.max_len = max_len
        self.embedd_dim = embedd_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encoding()

    def _build_positional_encoding(self):
        encodings = torch.zeros(
            self.max_len, self.embedd_dim, dtype=torch.float32
        )

        position_idx = torch.arange(
            0, self.max_len, dtype=torch.float32
        ).unsqueeze(-1)

        embed_skip_dim = torch.arange(
            0, self.embedd_dim, 2, dtype=torch.float32
        )

        div_term = 10000 ** (embed_skip_dim / self.embedd_dim)

        encodings[:, 0::2] = torch.sin(position_idx / div_term)
        encodings[:, 1::2] = torch.cos(position_idx / div_term)

        return nn.Parameter(encodings, requires_grad=self.requires_grad)

    def forward(self, x):
        return x + self.encodings[:x.shape[1]].unsqueeze(0)

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.embedding_dimension)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.embedding_dimension)

        self.src_positional_encoding = PositionalEncoding(config.src_sequence_length, config.embedding_dimension, config.learn_pos_embedd)
        self.tgt_positional_encoding = PositionalEncoding(config.tgt_sequence_length, config.embedding_dimension, config.learn_pos_embedd)

    def forward_src(self, input):
        embeddings = self.src_embeddings(input)
        return self.src_positional_encoding(embeddings)
    def forward_tgt(self, input):
        embeddings = self.tgt_embeddings(input)
        return self.tgt_positional_encoding(embeddings)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.embedding_dimension % config.num_attention_heads, "Embeddig dimention must divisible by number of heads"

        self.head_dim = config.embedding_dimension // config.num_attention_heads
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

    def forward(self, src, tgt = None, attention_mask = None, causal = False):
        batch, seq_len, embed = src.shape 
        if tgt is None:
            q = self.q_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, embed).trasnpose(1,2)
            k = self.k_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, embed).trasnpose(1,2)
            v = self.v_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, embed).trasnpose(1,2)

            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attetnion_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)
            attetnion_out = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, dropout_p= self.config.attention_dropout_p if self.training else 0.0)




if __name__ == "__main__":
    config = TransformerConfig()



