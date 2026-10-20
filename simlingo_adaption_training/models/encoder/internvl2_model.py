from torch import nn
from transformers import AutoModel

from simlingo_adaption_training.utils.internvl2_utils import resolve_internvl_variant_source


class LingoInternVLModel(nn.Module):
    def __init__(self, variant, *args, **kwargs):
        super().__init__()
        variant = resolve_internvl_variant_source(variant)
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True)
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except AttributeError:
            self.num_embeddings = self.model.language_model.vocab_size
        self.use_global_img = None
        self.processor = None
