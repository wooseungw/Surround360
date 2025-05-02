import torch
from torch import nn
from transformers import Blip2VisionModel, Blip2QFormerModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipImageProcessor, BlipTextProcessor
from transformers import Blip2Config
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


# class Blip2VisionModel(Blip2VisionModel):
#     def __init__(self, config):
#         super().__init__(config)
        
        
# class Blip2QFormerModel(Blip2QFormerModel):
#     def __init__(self, config):
#         super().__init__(config)
#         pass

# class Blip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    
#     def __init__(self, config: Blip2Config):
#         super().__init__(config)

#         self.vision_model = Blip2VisionModel(config.vision_config)

#         self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
#         self.qformer = Blip2QFormerModel(config.qformer_config)

#         self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
#         if config.use_decoder_only_language_model:
#             language_model = AutoModelForCausalLM.from_config(config.text_config)
#         else:
#             language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

#         # Update _tied_weights_keys using the base model used.
#         if language_model._tied_weights_keys is not None:
#             self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

#         self.language_model = language_model

#         # Initialize weights and apply final processing
#         self.post_init()

