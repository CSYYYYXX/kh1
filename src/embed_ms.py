import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PositionalEmbedding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = paddle.zeros((max_len, d_model))
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2).astype(paddle.float32)* -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistable=False)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]

class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding)
        nn.initializer.KaimingNormal()(self.tokenConv.weight)

    def forward(self, x):
        x = self.tokenConv(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        return x

class DataEmbedding(nn.Layer):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        value_emb = self.value_embedding(x)
        pos_emb = self.position_embedding(x)

        x = value_emb + pos_emb
        x = self.dropout(x)

        return x
