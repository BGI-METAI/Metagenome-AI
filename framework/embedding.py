from abc import ABC, abstractmethod
import torch
from torch.nn import Identity


class Embedding(ABC):
    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError()


class EsmEmbedding(Embedding):
    def __init__(self):
        model, alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm2_t12_35M_UR50D"
        )
        self.model = model
        self.model.contact_head = Identity()
        self.model.emb_layer_norm_after = Identity()
        self.model.lm_head = Identity()
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()

    def get_embedding(self, batch, pooling='cls'):
        data = [(fam, seq) for fam, seq in zip(batch["family"], batch["sequence"])]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            res = self.model(batch_tokens)
        # perform min max mean pool
        # Approach 1: Mean Pooling
        # pooled_encoder_output = torch.mean(enc_output, dim=1)
        # Approach 2: Using [CLS] 0th index
        # The first token of every sequence is always a special classification token ([CLS]).
        # The final hidden state corresponding to this token is used as the aggregate sequence representation
        # for classification tasks.
        if pooling == 'cls':
            res = res['logits'][:, 0, :]
        elif pooling == 'mean':
            pass
        elif pooling == 'max':
            pass
        else:
            raise NotImplementedError()
        return res

    def to(self, device):
        self.model.to(device)
