import math

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam

import lightning as L


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        learning_rate=0.0001,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        src = self.dropout(self.positional_encoding(self.src_embed(src)))

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)
        return src

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        tgt = self.dropout(self.positional_encoding(self.tgt_embed(tgt)))

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.output_layer(dec_output)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = None
        tgt_mask = None
        outputs = self(src, tgt_input, src_mask, tgt_mask)
        loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss 


    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = None
        tgt_mask = None

        outputs = self(src, tgt_input, src_mask, tgt_mask)
        loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = None
        tgt_mask = None

        outputs = self(src, tgt_input, src_mask, tgt_mask)
        loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
