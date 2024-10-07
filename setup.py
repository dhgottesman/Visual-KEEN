import torch
from overwritten_methods import llava_forward
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


MODEL_NAME_PATH = {
  "vicuna_7B": "lmsys/vicuna-7b-v1.5",
  "vicuna_13B": "lmsys/vicuna-13b-v1.5",
  "llava_7B": "llava-hf/llava-1.5-7b-hf",
  "llava_13B": "llava-hf/llava-1.5-13b-hf",
}

def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)

class VicunaModelAndTokenizer:
    def __init__(self, model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = tokenizer
        self.model = model
        self.model.cuda()
        self.num_layers = model.config.num_hidden_layers
        self.vocabulary_projection_function = lambda x, layer: self.model.lm_head(self.model.model.norm(x)) if layer < self.num_layers else self.model.lm_head(x) 

    def __repr__(self):
        """String representation of this class.
        """
        return (
            f"VicunaModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
            )

class LLaVAModelAndProcessor:
    def __init__(self, model_name):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            quantization_config=quantization_config, 
            # attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.num_layers = len(self.model.language_model.model.layers)
        self.vocabulary_projection_function = lambda x, layer: self.model.language_model.lm_head(self.model.language_model.model.norm(x)) if layer < self.num_layers else self.model.language_model.lm_head(x) 

        self.model.forward = llava_forward.__get__(self.model, self.model.__class__)

    def __repr__(self):
        """String representation of this class.
        """
        return (
            f"LLaVAModelAndProcessor(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"processor: {type(self.processor).__name__})"
            )


def setup(model_name, requires_grad=False):
    model_name = MODEL_NAME_PATH[model_name]
    if "vicuna" in model_name:
        mt = VicunaModelAndTokenizer(model_name)
    else:
        mt = LLaVAModelAndProcessor(model_name)
    mt.model.eval()
    set_requires_grad(requires_grad, mt.model)
    return mt