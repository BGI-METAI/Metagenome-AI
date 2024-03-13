#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/7/24 5:18 PM
# @Author  : zhangchao
# @File    : prottrans_embeddings.py
# @Email   : zhangchao5@genomics.cn
import re
import torch
from typing import Optional, Union, List

from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel, AlbertTokenizer, AlbertModel, \
    XLNetTokenizer, XLNetModel

from framework.embeddings import Embeddings
from framework.prottrans import PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, PROTTRANS_XLENT_TYPE, \
    POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE, POOLING_ALL_TYPE


class ProtTransEmbeddings(Embeddings):
    def __init__(
            self,
            model_name_or_path: str,
            mode: Optional[Union[PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, PROTTRANS_XLENT_TYPE]],
            **kwargs
    ):
        """
        ProtTrans Embedding models

        :param model_name_or_path (`str` or `os.PathLike`):
            - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
            Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
            user or organization name, like `dbmdz/bert-base-german-cased`.
            - A path to a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
            case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
            argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
            using the provided conversion scripts and loading the TensorFlow model afterwards.
        :param mode (`str`):
            Select the prottrans model to use. Support `PROTTRANS_T5_TYPE`, `PROTTRANS_BERT_TYPE`,
            `PROTTRANS_ALBERT_TYPE`, `PROTTRANS_XLENT_TYPE`
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.embedding_dims = None

        if mode == PROTTRANS_T5_TYPE:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = T5EncoderModel.from_pretrained(model_name_or_path).to(self.device)
        elif mode == PROTTRANS_BERT_TYPE:
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = BertModel.from_pretrained(model_name_or_path).to(self.device)
        elif mode == PROTTRANS_ALBERT_TYPE:
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = AlbertModel.from_pretrained(model_name_or_path).to(self.device)
        elif mode == PROTTRANS_XLENT_TYPE:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_name_or_path, **kwargs)
            self.model = XLNetModel.from_pretrained(model_name_or_path).to(self.device)
        else:
            raise ValueError(
                "Got an invalid `mode`, only support `PROTTRANS_T5_TYPE`, `PROTTRANS_BERT_TYPE`, `PROTTRANS_ALBERT_TYPE`, `PROTTRANS_XLENT_TYPE`")
        self.model.eval()

    @staticmethod
    def prepare_sequence(
            sequence: Optional[Union[str, List[str]]],
            add_separator: bool = True
    ) -> List[str]:
        """
        prepare protein sequence, add a space separator between each amino acid character and replace rare amino acids with `X`.

        :param sequence:
            amino acid sequences
        :param add_separator:
            whether to add space delimiters to each sequence. default is True.
        :return:
        """
        if isinstance(sequence, List):
            sequence = [re.sub(r'[UZOB]', 'X', seq) for seq in sequence]

        elif isinstance(sequence, str):
            sequence = [re.sub(r'[UZOB]', 'X', sequence)]
        else:
            raise ValueError('Error: Got an invalid protein sequence!')

        if add_separator:
            return [' '.join(list(seq)) for seq in sequence]
        else:
            return sequence

    @torch.no_grad()
    def get_embedding(
            self,
            protein_seq: Optional[Union[str, List[str]]],
            *,
            add_separator: bool = True,
            pooling: Optional[Union[POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE]],
            **kwargs
    ):
        """
        calculate protein sequence embeddings using the pLMs.

        :param protein_seq:
            the input amino acid sequences
        :param add_separator:
            whether to add space delimiters to each sequence. default is True.
        :param pooling:
            which method to be choose to generate the final protein sequence embeddings.
        :return:
        """

        protein_seq = self.prepare_sequence(sequence=protein_seq, add_separator=add_separator)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=protein_seq,
            add_special_tokens=True,
            padding='longest'
        )
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(self.device)
        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = embeddings.last_hidden_state
        self.embedding_dims = embeddings.size(2)

        if pooling == POOLING_CLS_TYPE:
            if self.mode in [PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE]:
                return embeddings[:, 0, :].detach().cpu().numpy()
            else:
                raise ValueError(
                    f'Error: `{POOLING_CLS_TYPE}` only support `{PROTTRANS_BERT_TYPE}` and `{PROTTRANS_ALBERT_TYPE}`')
        elif pooling == POOLING_MEAN_TYPE:
            return embeddings.mean(1).detach().cpu().numpy()
        elif pooling == POOLING_SUM_TYPE:
            return embeddings.sum(1).detach().cpu().numpy()
        elif pooling == POOLING_ALL_TYPE:
            return embeddings.detach().cpu().numpy()

    def get_embedding_dim(self):
        return self.embedding_dims

