"""DS-A oracle: a NumPy reference implementation of DeepSeek-V4.1-Flash's forward pass (see README.md)."""
from .config import Config, Mode, layer_modes
from .quant import QuantConfig
from .weights import DictWeights, GGUFWeights, Weights, load_from_gguf
from .model import Model, ModelCache, model_from_gguf

__all__ = ["Config", "Mode", "layer_modes", "QuantConfig", "DictWeights", "GGUFWeights", "Weights", "load_from_gguf",
           "Model", "ModelCache", "model_from_gguf"]
