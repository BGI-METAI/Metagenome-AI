import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """Module to convert original sentence into a vector of d_model sized embeddings (each word)"""

    def __init__(self, d_model: int, vocab_size: int):
        """_summary_

        Args:
            d_model (int): Dimension of the model, each word dimension
            vocab_size (int): How many words are there in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Precompute positional encodings and add them to word embeddings
    so the model has a sense of the position of each word
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """_summary_

        Args:
            d_model (int): Dimension of the positional encoding
            seq_len (int): Max length of the sentence, one vector for each position
            dropout (float): Prevent overfit
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create (seq_len, d_model) matrix
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) for all positions
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(positions * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Apply to all the sentences in a batch
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register it to the buffer of the model not as a parameter but to be saved when saving module
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to every word inside the sentence"""
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """Perform normalzation by calculating mean and std for each item in the batch
    and normalizing each item
    """

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        # nn.Parameter to make it learnable
        # Multiplicative
        self.alpha = nn.Parameter(torch.ones(1))
        # Additive
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class Classifier(nn.Module):
    def __init__(self, d_model, hidden_size, num_classes):
        super().__init__()
        self.d_model = d_model
        self.model = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """_summary_

        Args:
            d_model (int): Dimension of model
            h (int): Number of heads
            dropout (float): _description_
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) ->  (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # raw attention_scores used for visualization
        # return (attention_scores @ value), attention_scores
        return attention_scores @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # each head will see full sentence but smaller part of embedding
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Args:
            sublayer (_type_): Previous layer
        """
        # In the paper first applied is sublayer then norm
        # but this is more common in implementations
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            ResidualConnection(dropout) for _ in range(3)
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """_summary_

        Args:
            x: input of decoder
            encoder_output: output of encoder
            src_mask: mask of encoder (source language)
            tgt_mask: mask of decoder (target language)
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """Linear layer that is projecting embedding to the position in vocabulary"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        classifier: Classifier,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    num_classes: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 256,
) -> Transformer:
    """_summary_

    Args:
        src_vocab_size: _description_
        tgt_vocab_size: _description_
        src_seq_len: In case langauges are very different
        tgt_seq_len: both sizes do not have to be the same
        h: Number of heads. Defaults to 8.
        d_model: Embedding of the model. Defaults to 512.
        N: Number of encoder and decoder blocks. Defaults to 6.
        d_ff: Dimension of feed forward layer. Defaults to 2048.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    # Project from source to target language
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # Classifier used after pretraining, one hidden layer
    classifier = Classifier(d_model, int((d_model + num_classes) / 2), num_classes)

    # Create the transformer
    transformer = Transformer(
        encoder,
        decoder,
        classifier,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer,
    )

    # Init params with xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
