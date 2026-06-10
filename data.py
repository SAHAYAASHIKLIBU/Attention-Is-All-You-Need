import torch
from torch.utils.data import DataLoader
from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer
from datasets import load_from_disk

def translationCollater(src_tokenizer, tgt_tokenizer):

    def _collate_fn(batch):
        tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]
        src_ids = [torch.tensor(i["src_ids"]) for i in batch]

        src_pad_id = src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first = True, padding_value = src_pad_id)

        tgt_pad_id = tgt_tokenizer.special_tokens_dict["[PAD]"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first = True, padding_value = tgt_pad_id)

        return src_padded, tgt_padded
    return _collate_fn
