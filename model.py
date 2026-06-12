from sympy import sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm


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
    def __init__(self, config, requires_grad=False):
        super().__init__()
        self.config = config

        self.max_len = config.src_sequence_length
        self.embedd_dim = config.embedding_dimension
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

        self.src_positional_encoding = PositionalEncoding(config)
        self.tgt_positional_encoding = PositionalEncoding(config)

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

        assert config.embedding_dimension % config.num_attention_heads == 0, "Embeddig dimention must divisible by number of heads"

        self.head_dim = config.embedding_dimension // config.num_attention_heads
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

    def forward(self, src, tgt = None, attention_mask = None, causal = False):
        batch, seq_len, embed = src.shape 
        if tgt is None:
            q = self.q_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
            k = self.k_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
            v = self.v_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)

            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)
            attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, dropout_p= self.config.attention_dropout_p if self.training else 0.0, is_causal = causal)

        else:
            tgt_len = tgt.shape[-2]
            q = self.q_proj(tgt).reshape(batch, tgt_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
            k = self.k_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
            v = self.v_proj(src).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)

            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attetnion_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, tgt_len, 1)
            attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, dropout_p= self.config.attention_dropout_p if self.training else 0.0, is_causal = False)
        attention_out = attention_out.transpose(2, 1).flatten(2)
        attention_out = self.out_proj(attention_out)
        return attention_out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidded_size = config.embedding_dimension * config.mlp_ratio
        self.intermidiate_dense = nn.Linear(config.embedding_dimension, self.hidded_size)
        self.activation = nn.GELU()
        self.intermidiate_drop = nn.Dropout(config.hidden_dropout_p)
        self.output_dense = nn.Linear(self.hidded_size, config.embedding_dimension)
        self.output_drop = nn.Dropout(config.hidden_dropout_p)
    
    def forward(self, x):
        x = self.intermidiate_dense(x)
        x = self.activation(x)
        x = self.intermidiate_drop(x)
        x = self.output_dense(x)
        return self.output_drop(x)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_attention = Attention(config)
        self.feedforward = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layernorm = nn.LayerNorm(config.embedding_dimension)
        self.final_layernorm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, x, src_mask=None):
        x = x + self.dropout(self.enc_attention(x))
        x = self.layernorm(x)
        x = x + self.feedforward(x)
        return self.final_layernorm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.de_masked_attention = Attention(config)
        self.de_masked_attetnion_dropout = nn.Dropout(config.hidden_dropout_p)
        self.de_masked_norm = nn.LayerNorm(config.embedding_dimension)

        self.cross_attention = Attention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dimension)
        
        self.de_feedforward = FeedForward(config)
        self.final_layernorm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        x = tgt + self.de_masked_attetnion_dropout(self.de_masked_attention(tgt, causal = True))
        x = self.de_masked_norm(x)

        x = x + self.cross_attention_dropout(self.cross_attention(src, x, attention_mask = src_mask))
        x = self.cross_attention_layernorm(x)

        x = x + self.de_feedforward(x)
        return self.final_layernorm(x)



class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encodings = Embeddings(config)
        self.decoder = nn.ModuleList(
        [TransformerDecoder(config) for _ in range(config.decoder_depth)]
        )
        self.encoder = nn.ModuleList(
        [TransformerEncoder(config) for _ in range(config.encoder_depth)]
        )
        self.pred = nn.Linear(config.embedding_dimension, config.tgt_vocab_size)

        self.apply(_init_weights_)
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        src = self.encodings.forward_src(src)
        tgt = self.encodings.forward_tgt(tgt)
        for layer in self.encoder:
            src = layer(src, src_mask = src_mask)
        for layer in self.decoder:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        x = self.pred(tgt)
        return x

    @torch.no_grad()
    def inference(self, src, start_id, end_id, max_len= 512):
        tgt_ids = torch.tensor([start_id], device = src.device).unsqueeze(0)

        src_embeddings = self.encodings.forward_src(src)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
            
        for _ in tqdm(range(max_len)):
            tgt_embeddings = self.encodings.forward_tgt(tgt_ids)

            for layer in self.decoder:
                tgt_embeddings = layer(src_embeddings, tgt_embeddings)

            tgt_embeddings = tgt_embeddings[:, -1]
            pred = self.pred(tgt_embeddings)
            pred = pred.argmax(dim=-1).unsqueeze(0)
            if pred == end_id:
                break

            tgt_ids = torch.cat([tgt_ids, pred], dim = -1)
        return tgt_ids.squeeze(0).cpu().tolist()
    
def _init_weights_(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


            
            
            





if __name__ == "__main__":
    config = TransformerConfig()
    data = torch.randint(low = 0, high = 10000, size = (5, 88))
    data2 = torch.randint(low = 0, high = 10000, size = (5, 78))
    a = Transformer(config)
    english = torch.randint(low = 0, high = 10000, size = (1, 88))
    print(a.inference(english, 2, 3))


