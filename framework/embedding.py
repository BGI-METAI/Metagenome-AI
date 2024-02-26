from abc import ABC, abstractmethod
import torch
from torch.nn import Identity
import esm


class Embedding(ABC):
    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError()

    @abstractmethod
    def get_embedding_dim(self, batch=None, pooling="cls"):
        raise NotImplementedError()

    @abstractmethod
    def to(self):
        raise NotImplementedError()


class EsmEmbedding(Embedding):
    def __init__(self):
        # model, alphabet = torch.hub.load(
        #     "facebookresearch/esm:main", "esm2_t12_35M_UR50D"
        # )
        # model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = model
        self.alphabet = alphabet
        self.model.contact_head = Identity()
        self.model.emb_layer_norm_after = Identity()
        self.model.lm_head = Identity()
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.embed_dim = model.embed_dim

    def get_embedding(self, batch, pooling="cls"):
        data = [(fam, seq) for fam, seq in zip(batch["family"], batch["sequence"])]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            res = self.model(batch_tokens)

        # The first token of every sequence is always a special classification token ([CLS]).
        # The final hidden state corresponding to this token is used as the aggregate sequence representation
        # for classification tasks.
        if pooling == "cls":
            seq_repr = res["logits"][:, 0, :]
        elif pooling == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(res["logits"][i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        elif pooling == "max":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(res["logits"][i, 1 : tokens_len - 1].max(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError()
        return seq_repr

    def to(self, device):
        self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim
