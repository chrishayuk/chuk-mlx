import json
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

class ModelConfig(BaseModel):
    #architectures: Optional[List[str]]
    #attention_bias: Optional[float] = None
    #attention_dropout: Optional[float]
    #bos_token_id: Optional[int]
    #eos_token_id: Optional[int]
    hidden_act: str
    hidden_size: int
    #initializer_range: Optional[float]
    intermediate_size: int
    #max_position_embeddings: Optional[int]
    #model_type: Optional[str]
    num_attention_heads: Optional[int] = None
    num_hidden_layers: int
    num_key_value_heads: Optional[int] = None
    rms_norm_eps: Optional[float] = None

    # rope settings
    # not sure what scaling should be
    rope_scaling: Optional[str] = None
    rope_theta: Optional[float] = None
    rope_traditional: Optional[bool] = False
    #sliding_window: Optional[dict] = None
    #tie_word_embeddings: Optional[bool]
    #torch_dtype: Optional[str]
    #transformers_version: Optional[str]
    #use_cache: Optional[bool]
    vocab_size: int

    class Config:
        allow_population_by_field_name = True

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
        table = "Configurations:\n\n"
        max_key_length = max(len(str(key)) for key in self.dict().keys())
        max_value_length = max(len(str(value)) if not isinstance(value, list) else max(len(str(item)) for item in value) for value in self.dict().values())
        table += f"| {'Parameter':<{max_key_length}} | {'Value':<{max_value_length}} |\n"
        table += f"| {'-' * max_key_length} | {'-' * max_value_length} |\n"
        for key, value in self.dict().items():
            if isinstance(value, list):
                value_str = ', '.join(map(str, value))
            else:
                value_str = str(value)
            table += f"| {str(key):<{max_key_length}} | {value_str:<{max_value_length}} |\n"
        print(table)