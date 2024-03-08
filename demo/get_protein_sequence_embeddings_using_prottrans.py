#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:08 PM
# @Author  : zhangchao
# @File    : get_protein_sequence_embeddings_using_prottrans.py
# @Email   : zhangchao5@genomics.cn
from framework.prottrans import ProtTransEmbeddings, PROTTRANS_T5_TYPE, POOLING_MEAN_TYPE

if __name__ == '__main__':

    demo_sequences = [
        'MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPL',
        'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVE',
        'LIVGHFSGIKYKGEKAQASEVDVNKMCCWV'
    ]

    model_name_or_path = '/media/Data/zhangchao/metageomics/weights/prot_t5_xl_half_uniref50-enc'
    prottrans_model = ProtTransEmbeddings(
        model_name_or_path=model_name_or_path,
        mode=PROTTRANS_T5_TYPE,
        do_lower_case=False,
        legacy=False
    )

    protein_embed = prottrans_model.get_embedding(
        protein_seq=demo_sequences,
        add_separator=True,
        pooling=POOLING_MEAN_TYPE
    )

    print(protein_embed.shape)


