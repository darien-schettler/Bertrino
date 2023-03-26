from dataclasses import dataclass


@dataclass
class BertrinoConfig:
    batch_size: int = 256
    n_shuffle_buffer: int = 256 * 6
    seq_len: int = 129
    vocab_len: int = 5163 # +3 is for pad, mask, and cls (0 and 5161, 5162 respectively)
    pad_token: int = 0
    mask_token: int = 5161
    cls_token: int = 5162
    n_epochs: int = 100
    mlm_pct: float = 0.15
    lr: float = 0.0005
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout_rate: float = 0.1
    loss_fn: str="sparse_categorical_crossentropy"
    optimizer: str="adam"
