from abc import ABC, abstractmethod
import torch
from torch.nn import Identity


class Embedding(ABC):
    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError()


class EsmEmbedding(Embedding):
    def get_embedding(self, data=[]):
        model, alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm2_t30_150M_UR50D"
        )
        model.contact_head = Identity()
        model.emb_layer_norm_after = Identity()
        model.lm_head = Identity()
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        # make data as input
        data = [
            (
                "protein1",
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            ),
            (
                "protein2",
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            ),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            res = model(batch_tokens)
        return res
