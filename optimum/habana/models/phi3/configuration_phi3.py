from transformers.models.phi3.configuration_phi3 import Phi3Config

class Phi3Config(Phi3Config):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.43.0/src/transformers/models/phi3/configuration_phi3.py
    Changes:
    - added `_rope_scaling_adjustment` before calling `_rope_scaling_validation` (for backwards-compatibility)
    """
    def __init__(
        self,
        vocab_size=32064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=1,
        eos_token_id=32000,
        pad_token_id=32000,
        sliding_window=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sliding_window=sliding_window,
            **kwargs,
        )
