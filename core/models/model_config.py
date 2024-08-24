import json
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from pathlib import Path

class ModelConfig(BaseModel):
    # set the architexture
    architectures: Optional[List[str]] = None
    #attention_dropout: Optional[float]

    # special tokens
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]

    # hidden layers
    hidden_act: str
    
    hidden_size: int

    # activiation function
    hidden_act: Optional[str] = None
    hidden_activation: Optional[str] = None


    num_hidden_layers: int
    #initializer_range: Optional[float]
    intermediate_size: int
    #model_type: Optional[str]

    # tie word embeddings to layers
    tie_word_embeddings: Optional[bool] = True

    # attention settings
    num_attention_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: Optional[bool] = False
    rms_norm_eps: Optional[float] = None

    # rope settings
    max_position_embeddings: Optional[int]
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_theta: float = 10000
    rope_traditional: bool = False

    #sliding_window: Optional[dict] = None
    #torch_dtype: Optional[str]
    #transformers_version: Optional[str]
    #use_cache: Optional[bool]

    # MLP Bias
    mlp_bias: Optional[bool] = False
    
    # vocab
    vocab_size: int

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")
            
        # Handle Gemma's 'hidden_activation'
        if self.hidden_activation and not self.hidden_act:
            self.hidden_act = self.hidden_activation

    class ConfigDict:
        populate_by_name=True

    @classmethod
    def load(cls, file_path: Path) -> 'ModelConfig':
        if file_path.is_dir():
            file_path /= "config.json"
        with open(file_path, "r") as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    @classmethod
    def from_dict(cls, config_data) -> 'ModelConfig':
        # load the config from the dictionary
        return cls(**config_data)
    
    def print_layers(self):
        num_layers = self.num_hidden_layers
        
        table = f"""\
Loaded config.json
| Layer | Layer Name |
|-------|------------|
| 0     | Embeddings |
"""

        for layer in range(1, num_layers + 1):
            table += f"| {layer:<5} | Layer {layer:<2}   |\n"

        table += f"| {num_layers + 1:<5} | Output     |\n"
        print(table)

    def print_config(self):
        # Get a dictionary of all fields and their values
        config_data = self.dict()

        # Exclude parameters with None values or with values that are the same as their defaults
        filtered_data = {
            key: value for key, value in config_data.items()
            if value is not None and value != self.__fields__[key].default
        }

        # Prepare the output table
        if filtered_data:
            max_key_length = max(len(str(key)) for key in filtered_data.keys())
            max_value_length = max(len(str(value)) if not isinstance(value, list) else max(len(str(item)) for item in value) for value in filtered_data.values())

            table = "Configurations:\n\n"
            table += f"| {'Parameter':<{max_key_length}} | {'Value':<{max_value_length}} |\n"
            table += f"| {'-' * max_key_length} | {'-' * (max_value_length+1)} |\n"

            for key, value in filtered_data.items():
                if isinstance(value, list):
                    value_str = ', '.join(map(str, value))
                else:
                    value_str = str(value)
                table += f"| {str(key):<{max_key_length}} | {value_str:<{(max_value_length+1)}} |\n"

            print(table)
        else:
            print("No configurations to display.")
