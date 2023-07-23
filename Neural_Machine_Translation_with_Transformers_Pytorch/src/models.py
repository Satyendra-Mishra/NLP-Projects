import math
import torch
from torch import nn
from .embeddings import InputEmbeddings, PositionalEmbeddings


def build_transformer(src_vocab_size, tgt_vocab_size, seq_len=128, d_model=512, h=8, L_enc=6, L_dec=6, d_ff=2048, norm_first=False):
    # source embeddings
    src_input_embd = InputEmbeddings(src_vocab_size, d_model)
    src_pe = PositionalEmbeddings(seq_len, d_model, dropout_rate=0.3)
    # target embeddings
    tgt_input_embd = InputEmbeddings(tgt_vocab_size, d_model)
    tgt_pe = PositionalEmbeddings(seq_len, d_model, 0.3)
    # Encoder-Decoder
    encoder = TransformerEncoder(L=L_enc, d_model=d_model, h=h, d_ff=d_ff, dropout_rate=0.3, norm_first=norm_first)
    decoder = TransformerDecoder(L=L_dec, d_model=d_model, h=h, d_ff=d_ff, dropout_rate=0.3, norm_first=norm_first)
    # Trasnformers model
    transformer = Transformer(encoder, decoder, src_input_embd, src_pe, tgt_input_embd, tgt_pe)
    # initialize the parameters
    for parameter in transformer.parameters():
        if parameter.ndim > 1:
            nn.init.xavier_uniform_(parameter)
    return transformer



# implementing the transformers model
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_pe, tgt_embed, tgt_pe):
        super(Transformer, self).__init__()
        self.input_embeddings_src = src_embed
        self.positional_embeddings_src = src_pe
        self.input_embeddings_tgt = tgt_embed
        self.positional_embeddings_tgt = tgt_pe
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(tgt_embed.d_model, tgt_embed.vocab_size)

    def encode(self, x, src_mask):
        x = self.input_embeddings_src(x)
        x = self.positional_embeddings_src(x)
        return self.encoder(x, src_mask)

    def decode(self, x, memory, src_mask, tgt_mask):
        x = self.input_embeddings_tgt(x)
        x = self.positional_embeddings_tgt(x)
        return self.decoder(x, memory, src_mask, tgt_mask)
    
    def project(self, x):
        return self.linear(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder
        encoder_output = self.encode(src, src_mask)
        # decoder
        x = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        # projection
        x = self.project(x)
        return x
    

# Transformers Encoder and Decoder
class TransformerEncoder(nn.Module):
    def __init__(self, L: int, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout_rate, norm_first=norm_first) for _ in range(L)])

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, L: int, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout_rate, norm_first=norm_first) for _ in range(L)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.layer_norm(x)
 


# Transformer - one encoder block
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(2)])
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])
        self.norm_first = norm_first

    def forward(self, x, mask):
        if self.norm_first:     # Pre-LN
            # self attention
            attn_input = self.layer_norms[0](x)
            attn_output = self.self_attention_block(attn_input, attn_input, attn_input, mask)
            x = self.residual_connections[0](x, attn_output)
            # feedforward block
            ffn_input = self.layer_norms[1](x)
            ffn_output = self.feed_forward_block(ffn_input)
            x = self.residual_connections[1](x, ffn_output)
        else:
            attn_output = self.self_attention_block(x, x, x, mask)
            x = self.residual_connections[0](x, attn_output)
            x = self.layer_norms[0](x)
            # feedforward block
            ffn_output = self.feed_forward_block(x)
            x = self.residual_connections[1](x, ffn_output)
            x = self.layer_norms[1](x)
        return x


# Transformer - One decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(3)])
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])
        self.norm_first = norm_first

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        if self.norm_first:        # Pre-LN
            # masked self attention
            self_attn_input = self.layer_norms[0](x)
            self_attn_output = self.self_attention_block(self_attn_input, self_attn_input, self_attn_input, tgt_mask)
            x = self.residual_connections[0](x, self_attn_output)
            # cross attention
            cross_attn_input = self.layer_norms[1](x)
            cross_attn_output = self.cross_attention_block(cross_attn_input, encoder_output, encoder_output, src_mask)
            x = self.residual_connections[1](x, cross_attn_output)
            # feedforward block
            ffn_input = self.layer_norms[2](x)
            ffn_output = self.feed_forward_block(ffn_input)
            x = self.residual_connections[2](x, ffn_output)
        else:                      # Post-LN
            # masked self attention
            self_attn_output = self.self_attention_block(x, x, x, tgt_mask)
            x = self.residual_connections[0](x, self_attn_output)
            x = self.layer_norms[0](x)
            # cross attention
            cross_attn_output = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
            x = self.residual_connections[1](x, cross_attn_output)
            x = self.layer_norms[1](x)
            # feedforward block
            ffn_output = self.feed_forward_block(x)
            x = self.residual_connections[2](x, ffn_output)
            x = self.layer_norms[2](x)
        return x



# Multihead attention block
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout_rate: float):
        super(MultiheadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model must be divisible by num of heads"
        self.d_k = self.d_model // self.h
        # projection matrices for Q, K and V
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.WO = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def split_heads(self, mat):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        return mat.view(mat.shape[0], mat.shape[1], self.h, self.d_k).transpose(1,2)

    def concat_heads(self, mat):
        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k)
        mat = mat.transpose(1,2).contiguous()
        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h*d_k=d_model)
        return mat.view(mat.shape[0], -1, self.h*self.d_k)

    @staticmethod
    def scaled_dot_product_attention(queries, keys, values, mask, dropout: nn.Dropout=None):
        d_k = queries.shape[-1]
        # (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))/math.sqrt(d_k)
        # apply the mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0, 1e-9)
        # apply the softmax
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        # dropout
        if dropout:
            attention_scores = dropout(attention_scores)
        # return the results
        return torch.matmul(attention_scores, values), attention_scores

    def forward(self, Q, K, V, mask):
        Q_proj = self.WQ(Q) # (batch_size, seq_len, d_model)
        K_proj = self.WK(K) # (batch_size, seq_len, d_model)
        V_proj = self.WV(V) # (batch_size, seq_len, d_model)

        # split into feature subsets for different heads
        Q_proj = self.split_heads(Q_proj)
        K_proj = self.split_heads(K_proj)
        V_proj = self.split_heads(V_proj)
        # compute the self attention
        x, self.attention_scores = MultiheadAttentionBlock.scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask, dropout=self.dropout)
        # merge the output from each head
        x = self.concat_heads(x)

        return self.WO(x)


# feed forward block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)     # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)     # (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return x


# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias

    

# residual connections
class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, output):
        return x + self.dropout(output)
    


    


   



