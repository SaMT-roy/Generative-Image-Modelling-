import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rope(x, sin, cos):
    """
    x   : (B, h, T, d) even-sized last dim (d must be multiple of 2)
    sin : (T, d//2)     broadcastable
    cos : (T, d//2)
    """
    # This separates each feature vector's dimensions into 2 halves — like real and imaginary parts.
    x_even = x[..., 0::2]      # Get even-dimension values → shape: (B, h, T, d/2)
    x_odd  = x[..., 1::2]      # Get odd-dimension values → shape: (B, h, T, d/2)

    # This is a 2D rotation formula applied to each positional index and head.
    # It "rotates" the embedding vector in its dimensional space based on position.
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos
    
    # interleave even/odd back together
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)   # (..., d/2, 2)
    return x_rot.view(x.shape)                             # (..., d)

def make_sincos(seq_len, dim, base=10000):
    '''
    Returns sin, cos with shape (seq_len, dim//2)
    '''
    pos = torch.arange(seq_len, dtype=torch.float32)           # (T,)
    i = torch.arange(0, dim, 2, dtype=torch.float32) / dim    # (d/2,)
    theta = pos[:, None] / (base ** i[None, :])               # (T, d/2)
    return torch.sin(theta), torch.cos(theta)

class LoraLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8, alpha=32, name='lora'):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self._scale = alpha / rank

        # LoRA dense layers
        self.A = nn.Linear(in_features=input_dim, out_features=rank, bias=False)
        self.B = nn.Linear(in_features=rank, out_features=output_dim, bias=False)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.A.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        lora_output = self.B(self.A(x)) * self._scale
        return lora_output

class MultiHeadAttention(nn.Module):
    """
    Vanilla multi-head (scaled-dot-product) attention implemented from scratch.

    Args
    ----
    d_model     : int   – total embedding size (must be divisible by num_heads)  
    num_heads   : int   – number of attention heads  
    dropout     : float – dropout on attention weights (0.0 = no dropout)
    """
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections for Q, K, V and final output
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.lora_wq = LoraLayer(d_model, d_model, name='lora_wq')
        self.lora_wv = LoraLayer(d_model, d_model, name='lora_wv')

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, B):
        """
        Reshape (B, T, d_model) → (B, num_heads, T, depth)
        so we can run attention on each head in parallel.
        """
        x = x.view(B, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)  # (B, num_heads, T, depth)

    def _scaled_dot_product_attention(self, q, k, v, mask, dropout):
        """
        Core attention: softmax(QKᵀ / √d_k) V
        Returns: (B, h, T_q, depth_v)
        """
        dk = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))  # (B,h,T_q,T_k)

        if mask is not None:
            # broadcast automatically if mask rank < scores rank
            scores = scores.masked_fill(mask == 1, -1e9)  # large negative → zero probability

        attn = F.softmax(scores, dim=-1)
        attn = dropout(attn)
        output = torch.matmul(attn, v)  # (B,h,T_q,depth)
        return output

    def forward(self, query, value=None, key=None, mask=None, use_causal_mask=False):
        if value is None:
            value = query
        if key is None:
            key = value

        B = query.size(0)
        Tq = query.size(1)  # sequence length of Q
        Tk = key.size(1)

        # 1. Linear projections
        q = self.wq(query) + self.lora_wq(query)  # (B, T_q, d_model)
        k = self.wk(key)                          # (B, T_k, d_model)
        v = self.wv(value) + self.lora_wv(value)  # (B, T_v, d_model)

        # 2. Reshape for multi-head
        q = self._split_heads(q, B)  # (B, h, T_q, depth)
        k = self._split_heads(k, B)  # (B, h, T_k, depth)
        v = self._split_heads(v, B)  # (B, h, T_v, depth)

        # 3. ROTARY
        # Build sin/cos for the longest sequence we need this step
        max_len = max(Tq, Tk)
        sin, cos = make_sincos(max_len, self.depth)  # depth = d_model / num_heads

        # Slice sin/cos to actual lengths (broadcast works automatically)
        # RoPE modifies Q and K such that their dot product reflects not just content similarity but also relative position.
        q = apply_rope(q, sin[:Tq], cos[:Tq])  # rotate Q
        k = apply_rope(k, sin[:Tk], cos[:Tk])  # rotate K

        # 4. (Optional) Causal mask: block future positions
        if use_causal_mask:
            T_q = q.size(2)
            T_k = k.size(2)
            causal = torch.triu(torch.ones(T_q, T_k), diagonal=1).unsqueeze(0).unsqueeze(0)  # (1,1,T_q,T_k)
            if mask is None:
                mask = causal
            else:
                mask = torch.maximum(mask, causal)

        # 5. Scaled dot-product attention
        attn_out = self._scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 6. Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B,T_q,h,depth)
        attn_out = attn_out.view(B, -1, self.d_model)     # (B,T_q,d_model)

        # 7. Final linear layer
        output = self.wo(attn_out)  # (B,T_q,d_model)
        return output

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        norm_x = x / rms
        return norm_x * self.scale

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.rmsnorm = RMSNorm(d_model)

    def forward(self, x, padding_mask=None):
        rms_x1 = self.rmsnorm(x)
        attn_output = self.mha(
            query=rms_x1, value=rms_x1, key=rms_x1,
            mask=padding_mask,  # may be None
            use_causal_mask=True
        )
        rms_x1 = x + attn_output
        return rms_x1

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, factor=4):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, factor * hidden_dim, bias=False)  # W1
        self.lin2 = nn.Linear(factor * hidden_dim // 2, hidden_dim, bias=False)  # W2
        self.lora_lin1 = LoraLayer(hidden_dim, factor * hidden_dim, name='lora_lin1')  # LoRA to lin1

    def forward(self, x):
        x_ = self.lin1(x) + self.lora_lin1(x)  # shape: (..., 4d)
        a, b = torch.split(x_, x_.size(-1) // 2, dim=-1)  # split
        gated = a * (b * torch.sigmoid(b))  # SwiGLU: a ⊙ SiLU(b)
        return self.lin2(gated)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.swiglu = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.rmsnorm = RMSNorm(d_model)

    def forward(self, x):
        y = self.rmsnorm(x)
        y = self.swiglu(y)
        y = self.dropout(y)
        return x + y

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, d_model=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model)

    def forward(self, x, padding_mask=None):
        x = self.causal_self_attention(x, padding_mask=padding_mask)
        x = self.ffn(x)
        return x

class TokenEmbedding(nn.Module):
    """
    Args:
        vocab_size (int): Vocabulary size (number of unique tokens).
        d_model (int): Embedding dimension.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        """
        Args:
            x: Tensor of token indices → shape: (batch_size, sequence_length)
        
        Returns:
            Tensor of shape (batch_size, sequence_length, d_model)
        """
        return self.embedding(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, input_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.final_layer = nn.Linear(d_model, input_vocab_size)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            pad_mask = (~pad_mask).float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, padding_mask=pad_mask)

        x = self.rmsnorm(x)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits