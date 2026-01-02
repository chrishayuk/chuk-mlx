"""
Constants and enums for model families.

All model-type strings, architecture names, and other identifiers
are defined here to avoid magic strings throughout the codebase.
"""

from enum import Enum


class HFModelType(str, Enum):
    """HuggingFace model_type values from config.json."""

    # Llama family
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    CODELLAMA = "codellama"

    # Llama 4
    LLAMA4 = "llama4"

    # Gemma family
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    GEMMA3_TEXT = "gemma3_text"

    # Granite family
    GRANITE = "granite"
    GRANITE_MOE_HYBRID = "granitemoehybrid"

    # Jamba
    JAMBA = "jamba"

    # Mamba
    MAMBA = "mamba"

    # StarCoder2
    STARCODER2 = "starcoder2"

    # Qwen family
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"

    # GPT-2 family
    GPT2 = "gpt2"
    GPT_NEO = "gpt_neo"
    GPT_NEOX = "gpt_neox"

    # GPT-OSS (OpenAI open source MoE)
    GPT_OSS = "gpt_oss"


class HFArchitecture(str, Enum):
    """HuggingFace architecture class names from config.json."""

    # Llama family
    LLAMA_FOR_CAUSAL_LM = "LlamaForCausalLM"
    MISTRAL_FOR_CAUSAL_LM = "MistralForCausalLM"
    MIXTRAL_FOR_CAUSAL_LM = "MixtralForCausalLM"

    # Llama 4
    LLAMA4_FOR_CAUSAL_LM = "Llama4ForCausalLM"
    LLAMA4_FOR_CONDITIONAL_GENERATION = "Llama4ForConditionalGeneration"

    # Gemma family
    GEMMA_FOR_CAUSAL_LM = "GemmaForCausalLM"
    GEMMA2_FOR_CAUSAL_LM = "Gemma2ForCausalLM"
    GEMMA3_FOR_CAUSAL_LM = "Gemma3ForCausalLM"  # Text-only
    GEMMA3_FOR_CONDITIONAL_GENERATION = "Gemma3ForConditionalGeneration"  # VLM (not native)
    PALIGEMMA_FOR_CONDITIONAL_GENERATION = "PaliGemmaForConditionalGeneration"

    # Granite family
    GRANITE_FOR_CAUSAL_LM = "GraniteForCausalLM"
    GRANITE_MOE_HYBRID_FOR_CAUSAL_LM = "GraniteMoeHybridForCausalLM"

    # Jamba
    JAMBA_FOR_CAUSAL_LM = "JambaForCausalLM"

    # Mamba
    MAMBA_FOR_CAUSAL_LM = "MambaForCausalLM"

    # StarCoder2
    STARCODER2_FOR_CAUSAL_LM = "Starcoder2ForCausalLM"

    # Qwen family
    QWEN2_FOR_CAUSAL_LM = "Qwen2ForCausalLM"
    QWEN3_FOR_CAUSAL_LM = "Qwen3ForCausalLM"

    # GPT-2 family
    GPT2_LM_HEAD_MODEL = "GPT2LMHeadModel"
    GPT_NEO_FOR_CAUSAL_LM = "GPTNeoForCausalLM"
    GPT_NEOX_FOR_CAUSAL_LM = "GPTNeoXForCausalLM"

    # GPT-OSS (OpenAI open source MoE)
    GPT_OSS_FOR_CAUSAL_LM = "GptOssForCausalLM"


class DefaultVocabSize(int, Enum):
    """Default vocabulary sizes for different model families."""

    LLAMA2 = 32000
    LLAMA3 = 128256
    GEMMA = 256000
    GEMMA3 = 262144
    GPT2 = 50257
    STARCODER2 = 49152
    JAMBA = 65536
    MAMBA = 50280
    QWEN = 151936


class DefaultPositionEmbeddings(int, Enum):
    """Default max position embeddings for different model families."""

    GPT2 = 1024
    LLAMA2 = 4096
    LLAMA3 = 8192
    GEMMA = 8192
    GEMMA3 = 32768
    STARCODER2 = 16384
    JAMBA = 262144
    MAMBA = 2048


class DefaultRoPETheta(float, Enum):
    """Default RoPE theta values for different model families."""

    LLAMA2 = 10000.0
    LLAMA3 = 500000.0
    GEMMA3 = 1000000.0
    STARCODER2 = 100000.0


class DefaultNormEps(float, Enum):
    """Default normalization epsilon values."""

    LLAMA = 1e-5
    GEMMA = 1e-6
    GPT2 = 1e-5
    JAMBA = 1e-6
    MAMBA = 1e-5


class SpecialTokenId(int, Enum):
    """Common special token IDs."""

    # GPT-2 style
    GPT2_EOS = 50256
    GPT2_BOS = 50256

    # Llama style
    LLAMA_BOS = 1
    LLAMA_EOS = 2

    # Gemma style
    GEMMA_END_OF_TURN = 106


# Config field names (to avoid typos)
class ConfigField(str, Enum):
    """Field names in HuggingFace config.json."""

    MODEL_TYPE = "model_type"
    ARCHITECTURES = "architectures"
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_HIDDEN_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"
    NUM_KEY_VALUE_HEADS = "num_key_value_heads"
    INTERMEDIATE_SIZE = "intermediate_size"
    MAX_POSITION_EMBEDDINGS = "max_position_embeddings"
    ROPE_THETA = "rope_theta"
    RMS_NORM_EPS = "rms_norm_eps"
    TIE_WORD_EMBEDDINGS = "tie_word_embeddings"
    BOS_TOKEN_ID = "bos_token_id"
    EOS_TOKEN_ID = "eos_token_id"
    PAD_TOKEN_ID = "pad_token_id"
    SLIDING_WINDOW = "sliding_window"

    # GPT-2 specific
    N_EMBD = "n_embd"
    N_LAYER = "n_layer"
    N_HEAD = "n_head"
    N_INNER = "n_inner"
    N_POSITIONS = "n_positions"
    LAYER_NORM_EPSILON = "layer_norm_epsilon"

    # Gemma specific
    HEAD_DIM = "head_dim"
    SLIDING_WINDOW_PATTERN = "sliding_window_pattern"
    QUERY_PRE_ATTN_SCALAR = "query_pre_attn_scalar"

    # Jamba specific
    ATTN_LAYER_PERIOD = "attn_layer_period"
    EXPERT_LAYER_PERIOD = "expert_layer_period"
    NUM_EXPERTS = "num_experts"
    NUM_EXPERTS_PER_TOK = "num_experts_per_tok"
    MAMBA_D_STATE = "mamba_d_state"
    MAMBA_D_CONV = "mamba_d_conv"
    MAMBA_EXPAND = "mamba_expand"
