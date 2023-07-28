import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = (B, 1, L, L)
        self._mask = paddle.triu(paddle.ones(mask_shape), diagonal=1).astype(paddle.bool)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = paddle.triu(paddle.ones((L, scores.shape[-1])), diagonal=1).astype(paddle.bool)
        _mask_ex = _mask.unsqueeze(0).unsqueeze(0).expand((B, H, L, scores.shape[-1]))
        indicator = _mask_ex[np.arange(B)[:, None, None],
                    np.arange(H)[None, :, None],
                    index, :]
        self._mask = indicator.reshape(scores.shape)

    @property
    def mask(self):
        return self._mask
