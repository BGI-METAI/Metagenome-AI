import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]
        family = torch.tensor(src_target_pair["le_label"])

        # Encode text and get array of ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # Use tokenizer of source to encode target text because encoder learned attention on it
        enc_class_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        # Pad to reach seq_len, -2 enc because of SOS and EOS, -1 dec SOS
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # seq_len must be able to represent all sentences
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        enc_class_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_class_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # Expected output from decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert enc_class_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # 1. unsq to add seq dim, 2. unsq to add batch dim
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(decoder_input.size(0)),
            # (1, seq_len) * (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            # "src_text": src_text,  # for viz
            # "tgt_text": tgt_text,  # for viz
            "enc_class_input": enc_class_input,
            "enc_class_mask": (enc_class_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "family": family,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
