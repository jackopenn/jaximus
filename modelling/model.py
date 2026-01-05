import jax
from jax import numpy as jnp
from flax import nnx
from modelling.layers.core import MLP, GLU, Attention, create_norm
from modelling.layers.init import get_initializers
from parallel import logical_to_physical, shard_init


class Layer(nnx.Module):
    def __init__(
            self,
            hidden_dim,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_dim,
            act_fn,
            norm_type,
            norm_position,
            norm_epsilon,
            mlp_type,
            attn_use_bias,
            mlp_use_bias,
            rope_theta,
            qk_norm,
            qk_norm_type,
            qk_norm_epsilon,
            sliding_window,
            dtype,
            inits,
            rngs,
    ):
        super().__init__()
        self.norm_position = norm_position
        
        self.attention = Attention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qk_norm_epsilon=qk_norm_epsilon,
            use_bias=attn_use_bias,
            dtype=dtype,
            sliding_window=sliding_window,
            inits=inits,
            rngs=rngs,
        )
        
        self.ln_1 = create_norm(
            norm_type=norm_type,
            num_features=hidden_dim,
            epsilon=norm_epsilon,
            use_bias=attn_use_bias,
            rngs=rngs,
        )
        
        MLPFactory = MLP if mlp_type == "mlp" else GLU
        self.mlp = MLPFactory(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=act_fn,
            use_bias=mlp_use_bias,
            dtype=dtype,
            inits=inits,
            rngs=rngs,
        )
        
        self.ln_2 = create_norm(
            norm_type=norm_type,
            num_features=hidden_dim,
            epsilon=norm_epsilon,
            use_bias=mlp_use_bias,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        if self.norm_position == "pre":
            x = x + self.attention(self.ln_1(x), mask=mask)
            x = x + self.mlp(self.ln_2(x))
        else:  # post-norm
            x = self.ln_1(x + self.attention(x, mask=mask))
            x = self.ln_2(x + self.mlp(x))
        return x


class Model(nnx.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_dim,
        max_seq_len,
        norm_type,
        norm_position,
        norm_epsilon,
        mlp_type,
        act_fn,
        attn_use_bias,
        mlp_use_bias,
        lm_head_use_bias,
        qk_norm,
        qk_norm_type,
        qk_norm_epsilon,
        sliding_window,
        position_embedding_type,
        rope_theta,
        tie_word_embeddings,
        init_strategy,
        softcap,
        dtype,
        rngs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.position_embedding_type = position_embedding_type
        self.tie_word_embeddings = tie_word_embeddings
        self.softcap = softcap

        # used to estimate model flops 
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        dtype = getattr(jnp, dtype)
        act_fn = (lambda x: jnp.square(jax.nn.relu(x))) if act_fn == "relu_squared" else getattr(jax.nn, act_fn)
        inits = get_initializers(init_strategy, hidden_dim)

        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_dim,
            dtype=dtype,
            embedding_init=shard_init(inits["embed"], ("vocab", "embed")),
            rngs=rngs,
        )
        
        if position_embedding_type == "learned":
            self.pos_embedding = nnx.Embed(
                num_embeddings=max_seq_len,
                features=hidden_dim,
                dtype=dtype,
                embedding_init=shard_init(inits["embed"], ("seq", "embed")),
                rngs=rngs,
            )
        
        self.layers = nnx.List([
            Layer(
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                act_fn=act_fn,
                norm_type=norm_type,
                norm_position=norm_position,
                norm_epsilon=norm_epsilon,
                mlp_type=mlp_type,
                attn_use_bias=attn_use_bias,
                mlp_use_bias=mlp_use_bias,
                rope_theta=rope_theta if position_embedding_type == "rope" else None,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qk_norm_epsilon=qk_norm_epsilon if qk_norm else None,
                sliding_window=sliding_window,
                dtype=dtype,
                inits=inits,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = create_norm(
            norm_type=norm_type,
            num_features=hidden_dim,
            epsilon=norm_epsilon,
            use_bias=attn_use_bias,
            rngs=rngs,
        )
        
        if not tie_word_embeddings:
            self.lm_head = nnx.Linear(
                in_features=hidden_dim,
                out_features=vocab_size,
                use_bias=lm_head_use_bias,
                kernel_init=shard_init(inits["lm_head"], ("vocab", "embed")),
                bias_init=shard_init(inits["bias"], ("vocab", )),
                dtype=dtype,
                rngs=rngs,
            )

    def __call__(self, x, mask=None):
        x = self.token_embedding(x)

        if self.position_embedding_type == "learned":
            x = x + self.pos_embedding(jnp.arange(x.shape[1]))

        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)

        logits = self.lm_head(x, out_sharding=logical_to_physical(("batch", "seq", "vocab"))) if self.lm_head else self.token_embedding.attend(x)

        if self.softcap:
            logits = self.softcap * jnp.tanh(logits.astype(jnp.float32) / self.softcap)
            
        return logits
