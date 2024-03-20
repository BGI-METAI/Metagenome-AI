#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/7/24 5:18 PM
# @Author  : zhangchao
# @File    : prottrans_embeddings.py
# @Email   : zhangchao5@genomics.cn
import torch
from typing import Optional, Union
from transformers import T5EncoderModel, BertModel, AlbertModel, XLNetModel

from framework.embeddings import Embeddings
from framework.prottrans import PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, PROTTRANS_XLENT_TYPE, \
    POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE, POOLING_ALL_TYPE


class ProtTransEmbeddings(Embeddings):
    def __init__(
            self,
            model_name_or_path: str,
            mode_type: Optional[Union[PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, PROTTRANS_XLENT_TYPE]],
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

        self.mode_type = mode_type

        if mode_type == PROTTRANS_T5_TYPE:
            self.model = T5EncoderModel.from_pretrained(model_name_or_path)
        elif mode_type == PROTTRANS_BERT_TYPE:
            self.model = BertModel.from_pretrained(model_name_or_path)
        elif mode_type == PROTTRANS_ALBERT_TYPE:
            self.model = AlbertModel.from_pretrained(model_name_or_path)
        elif mode_type == PROTTRANS_XLENT_TYPE:
            self.model = XLNetModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(
                "Got an invalid `mode`, only support `PROTTRANS_T5_TYPE`, `PROTTRANS_BERT_TYPE`, `PROTTRANS_ALBERT_TYPE`, `PROTTRANS_XLENT_TYPE`")
        self.model.eval()

    @torch.no_grad()
    def get_embedding(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *,
            pooling: Optional[
                Union[POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE, POOLING_ALL_TYPE]] = POOLING_ALL_TYPE,
            convert2numpy: bool = False
    ):
        """
        calculate protein sequence embeddings using the pLMs.

        :param input_ids:
            the input amino acid sequences
        :param attention_mask:
        :param pooling:
            which method to be choose to generate the final protein sequence embeddings.
        :param convert2numpy:
            whether to convert the protein embedding to numpy format
        :return:
        """
        embeddings = self.model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
        embeddings = embeddings.last_hidden_state

        if pooling == POOLING_CLS_TYPE:
            if self.mode_type in [PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE]:
                result = embeddings[:, 0, :]
            else:
                raise ValueError(
                    f'Error: `{POOLING_CLS_TYPE}` only support `{PROTTRANS_BERT_TYPE}` and `{PROTTRANS_ALBERT_TYPE}`')
        elif pooling == POOLING_MEAN_TYPE:
            result = embeddings.mean(1)
        elif pooling == POOLING_SUM_TYPE:
            result = embeddings.sum(1)
        elif pooling == POOLING_ALL_TYPE:
            result = embeddings
        else:
            raise ValueError('Invalid output!')

        del embeddings, input_ids, attention_mask
        torch.cuda.empty_cache()

        if convert2numpy:
            return result.detach().cpu().numpy()
        else:
            return result

    @property
    def get_embedding_dim(self):
        if self.mode_type == PROTTRANS_T5_TYPE:
            return 1024
        elif self.mode_type == PROTTRANS_BERT_TYPE:
            return 1024

    def cuda(self):
        self.model.cuda()
