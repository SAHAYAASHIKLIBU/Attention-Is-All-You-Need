import torch
import os
from torch.utils.data import DataLoader
from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

def translationCollater(src_tokenizer, tgt_tokenizer):

    def _collate_fn(batch):
        tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]
        src_ids = [torch.tensor(i["src_ids"]) for i in batch]

        src_pad_id = src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first = True, padding_value = src_pad_id)
        src_pad_mask = (src_padded != src_pad_id)
        

        tgt_pad_id = tgt_tokenizer.special_tokens_dict["[PAD]"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first = True, padding_value = tgt_pad_id)

        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()
        input_tgt_mask = (input_tgt != tgt_pad_id)
        output_tgt[output_tgt == tgt_pad_id] = -100
        batch = {"src_input_ids": src_padded, 
                 "src_pad_mask": src_pad_mask, 
                 "tgt_input_ids": input_tgt, 
                 "tgt_pad_mask": input_tgt_mask, 
                 "tgt_outputs": output_tgt}
        
        return batch

        return src_padded, tgt_padded
    return _collate_fn


if __name__ == "__main__":

    path_to_data = os.getcwd() + "/tokenized_english2french_corpus"
    dataset = load_from_disk(path_to_data)
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tgt_tokenizer = FrenchTokenizer(os.getcwd() + "/french_wp.json")

    loader = DataLoader(dataset['train'], batch_size = 2, collate_fn = translationCollater(src_tokenizer, tgt_tokenizer), shuffle = True, num_workers = 128)
    for i in tqdm(loader):
        pass