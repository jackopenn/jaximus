# Jaximus

A JAX/Flax transformer implementation with configurable sharding.

## Transformer Data Flow

The diagram below shows the flow through the transformer with tensor shapes using logical axes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│                         (batch, seq)                                        │
│                          token IDs                                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOKEN EMBEDDING                                      │
│                    (batch, seq) → (batch, seq, embed)                       │
│                     weight: (vocab, embed)                                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   POSITION EMBEDDING (optional)                             │
│                       (batch, seq, embed)                                   │
│              + learned pos embed weight: (seq, embed)                       │
│                    or RoPE applied in attention                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         TRANSFORMER LAYER × N                               ┃
┃                                                                             ┃
┃  ┌────────────────────────────────────────────────────────────────────┐     ┃
┃  │ input: (batch, seq, embed)                                         │     ┃
┃  └──────────────────────────────┬─────────────────────────────────────┘     ┃
┃                                 │                                           ┃
┃                    ┌────────────┴────────────┐                              ┃
┃                    │                         │                              ┃
┃                    ▼                         │ (residual)                   ┃
┃  ┌─────────────────────────────────┐         │                              ┃
┃  │      PRE-NORM (ln_1)            │         │                              ┃
┃  │    (batch, seq, embed)          │         │                              ┃
┃  └────────────────┬────────────────┘         │                              ┃
┃                   │                          │                              ┃
┃                   ▼                          │                              ┃
┃  ┌─────────────────────────────────────────────────────────────────┐        ┃
┃  │                      ATTENTION                                  │        ┃
┃  │                                                                 │        ┃
┃  │  input: (batch, seq, embed)                                     │        ┃
┃  │                   │                                             │        ┃
┃  │       ┌───────────┼───────────┐                                 │        ┃
┃  │       ▼           ▼           ▼                                 │        ┃
┃  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                            │        ┃
┃  │  │ Q proj  │ │ K proj  │ │ V proj  │                            │        ┃
┃  │  └────┬────┘ └────┬────┘ └────┬────┘                            │        ┃
┃  │       │           │           │                                 │        ┃
┃  │       ▼           ▼           ▼                                 │        ┃
┃  │   (batch,     (batch,     (batch,                               │        ┃
┃  │    seq,        seq,        seq,                                 │        ┃
┃  │    heads,      kv_heads,   kv_heads,                            │        ┃
┃  │    head_dim)   head_dim)   head_dim)                            │        ┃
┃  │       │           │           │                                 │        ┃
┃  │       ▼           ▼           │                                 │        ┃
┃  │  ┌─────────┐ ┌─────────┐      │                                 │        ┃
┃  │  │ Q norm  │ │ K norm  │      │  (optional QK norm)             │        ┃
┃  │  └────┬────┘ └────┬────┘      │                                 │        ┃
┃  │       │           │           │                                 │        ┃
┃  │       ▼           ▼           │                                 │        ┃
┃  │  ┌─────────┐ ┌─────────┐      │                                 │        ┃
┃  │  │  RoPE   │ │  RoPE   │      │  (optional rotary embeddings)   │        ┃
┃  │  └────┬────┘ └────┬────┘      │                                 │        ┃
┃  │       │           │           │                                 │        ┃
┃  │       └───────────┴───────────┘                                 │        ┃
┃  │                   │                                             │        ┃
┃  │                   ▼                                             │        ┃
┃  │       ┌───────────────────────┐                                 │        ┃
┃  │       │  DOT PRODUCT ATTN     │                                 │        ┃
┃  │       │  Q @ K.T / √d → softmax → @ V                           │        ┃
┃  │       │  (batch, seq, heads, head_dim)                          │        ┃
┃  │       └───────────┬───────────┘                                 │        ┃
┃  │                   │                                             │        ┃
┃  │                   ▼                                             │        ┃
┃  │          ┌────────────────┐                                     │        ┃
┃  │          │    O proj      │                                     │        ┃
┃  │          │ (heads, head_dim) → (embed)                          │        ┃
┃  │          └───────┬────────┘                                     │        ┃
┃  │                  │                                              │        ┃
┃  │  output: (batch, seq, embed)                                    │        ┃
┃  └──────────────────┬──────────────────────────────────────────────┘        ┃
┃                     │                                                       ┃
┃                     ▼                                                       ┃
┃            ┌────────────────┐                                               ┃
┃            │   + residual   │ ◄────────────────────────────┘                ┃
┃            │ (batch, seq, embed)                                            ┃
┃            └───────┬────────┘                                               ┃
┃                    │                                                        ┃
┃       ┌────────────┴────────────┐                                           ┃
┃       │                         │                                           ┃
┃       ▼                         │ (residual)                                ┃
┃  ┌─────────────────────────────────┐                                        ┃
┃  │      PRE-NORM (ln_2)            │                                        ┃
┃  │    (batch, seq, embed)          │                                        ┃
┃  └────────────────┬────────────────┘                                        ┃
┃                   │                 │                                       ┃
┃                   ▼                 │                                       ┃
┃  ┌──────────────────────────────────────────────────────────────────┐       ┃
┃  │                         MLP                                      │       ┃
┃  │                                                                  │       ┃
┃  │  input: (batch, seq, embed)                                      │       ┃
┃  │                   │                                              │       ┃
┃  │                   ▼                                              │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │    up_proj     │                                      │       ┃
┃  │          │ (embed) → (intermediate)                              │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │     (batch, seq, intermediate)                                   │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │    act_fn      │  (e.g. ReLU², SiLU, GELU)            │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │     (batch, seq, intermediate)                                   │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │   down_proj    │                                      │       ┃
┃  │          │ (intermediate) → (embed)                              │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │  output: (batch, seq, embed)                                     │       ┃
┃  └──────────────────┬───────────────────────────────────────────────┘       ┃
┃                     │                                                       ┃
┃  ═══════════════════╪═══════════════════════════════════════════════════    ┃
┃  │ OR (GLU variant) │                                                       ┃
┃  ═══════════════════╪═══════════════════════════════════════════════════    ┃
┃                     │                                                       ┃
┃  ┌──────────────────────────────────────────────────────────────────┐       ┃
┃  │                         GLU                                      │       ┃
┃  │                                                                  │       ┃
┃  │  input: (batch, seq, embed)                                      │       ┃
┃  │                   │                                              │       ┃
┃  │                   ▼                                              │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │  up_gate_proj  │  (fused up + gate)                   │       ┃
┃  │          │ (embed) → (2 * intermediate)                          │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │     (batch, seq, 2 * intermediate)                               │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │     split      │                                      │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │           ┌──────┴──────┐                                        │       ┃
┃  │           ▼             ▼                                        │       ┃
┃  │         up            gate                                       │       ┃
┃  │    (batch, seq,   (batch, seq,                                   │       ┃
┃  │     intermediate)  intermediate)                                 │       ┃
┃  │           │             │                                        │       ┃
┃  │           │             ▼                                        │       ┃
┃  │           │     ┌────────────────┐                               │       ┃
┃  │           │     │    act_fn      │                               │       ┃
┃  │           │     └───────┬────────┘                               │       ┃
┃  │           │             │                                        │       ┃
┃  │           └──────┬──────┘                                        │       ┃
┃  │                  ▼                                               │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │   up * gate    │  (element-wise multiply)             │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │     (batch, seq, intermediate)                                   │       ┃
┃  │                  │                                               │       ┃
┃  │                  ▼                                               │       ┃
┃  │          ┌────────────────┐                                      │       ┃
┃  │          │   down_proj    │                                      │       ┃
┃  │          │ (intermediate) → (embed)                              │       ┃
┃  │          └───────┬────────┘                                      │       ┃
┃  │                  │                                               │       ┃
┃  │  output: (batch, seq, embed)                                     │       ┃
┃  └──────────────────┬───────────────────────────────────────────────┘       ┃
┃                     │                                                       ┃
┃                     ▼                                                       ┃
┃            ┌────────────────┐                                               ┃
┃            │   + residual   │ ◄────────────────────────────┘                ┃
┃            │ (batch, seq, embed)                                            ┃
┃            └───────┬────────┘                                               ┃
┃                    │                                                        ┃
┃  output: (batch, seq, embed)                                                ┃
┗━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FINAL NORM (ln_f)                                     │
│                      (batch, seq, embed)                                    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LM HEAD                                            │
│                  (batch, seq, embed) → (batch, seq, vocab)                  │
│                      weight: (vocab, embed)                                 │
│                  or tied: token_embedding.attend(x)                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SOFTCAP (optional)                                     │
│                   softcap * tanh(logits / softcap)                          │
│                      (batch, seq, vocab)                                    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT                                          │
│                       (batch, seq, vocab)                                   │
│                           logits                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Logical Axes Summary

| Axis           | Description                                    |
|----------------|------------------------------------------------|
| `batch`        | Batch dimension                                |
| `seq`          | Sequence length                                |
| `embed`        | Hidden/embedding dimension                     |
| `vocab`        | Vocabulary size                                |
| `heads`        | Number of attention heads                      |
| `kv_heads`     | Number of key/value heads (for GQA/MQA)        |
| `head_dim`     | Dimension per attention head                   |
| `intermediate` | MLP intermediate dimension (typically 4×embed) |

## Weight Shapes

| Layer              | Weight Shape                        | Logical Axes                    |
|--------------------|-------------------------------------|---------------------------------|
| token_embedding    | (vocab_size, hidden_dim)            | (vocab, embed)                  |
| pos_embedding      | (max_seq_len, hidden_dim)           | (seq, embed)                    |
| q_proj             | (hidden_dim, heads, head_dim)       | (embed, heads, head_dim)        |
| k_proj             | (hidden_dim, kv_heads, head_dim)    | (embed, heads, head_dim)        |
| v_proj             | (hidden_dim, kv_heads, head_dim)    | (embed, heads, head_dim)        |
| o_proj             | (heads, head_dim, hidden_dim)       | (heads, head_dim, embed)        |
| mlp.up_proj        | (hidden_dim, intermediate_dim)      | (embed, intermediate)           |
| mlp.down_proj      | (intermediate_dim, hidden_dim)      | (intermediate, embed)           |
| glu.up_gate_proj   | (hidden_dim, 2 * intermediate_dim)  | (embed, intermediate)           |
| glu.down_proj      | (intermediate_dim, hidden_dim)      | (intermediate, embed)           |
| lm_head            | (hidden_dim, vocab_size)            | (embed, vocab)                  |
