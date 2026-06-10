import os
import glob
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
import argparse

from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer

def build_english2french_dataset(path_to_data_root, 
                                 path_to_save, 
                                 test_prop=0.005,
                                 cache_dir=None):
    def _scentence_pair(path_to_data_root):
        files = [i for i in glob.glob(path_to_data_root+"/data/*.fr")]
        for file in files:
            with open(file, 'r', encoding = 'utf-8') as french, open(file[:-2]+"en", 'r', encoding = 'utf-8') as english:
                for fren, eng in zip(french, english):
                    yield {
                        "english_src" : eng.rstrip(),
                        "french_tgt" : fren.rstrip()
                    }
    dataset = Dataset.from_generator(
    _scentence_pair,
    gen_kwargs={"path_to_data_root": path_to_data_root}
)
    
    hf_dataset = dataset.train_test_split(test_size=test_prop)

    hf_dataset.save_to_disk(path_to_save)

def tokenize_english2french_dataset(path_to_hf_data, 
                                    path_to_save, 
                                    num_workers=24, 
                                    truncate=False, 
                                    max_length=512, 
                                    min_length=5):

    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json", truncate=truncate, max_length=max_length)
    english_tokenzer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    raw_dataset = load_from_disk(path_to_hf_data)
    
    def _tokenize_text(examples):

        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        src_ids = english_tokenzer(english_text, truncation=True, max_length=512)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)

        batch = {"src_ids": src_ids, 
                 "tgt_ids": tgt_ids}
        
        return batch
    
    tokenized_dataset = raw_dataset.map(_tokenize_text, batched=True, num_proc=num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])

    filter_func = lambda batch: [len(e) > min_length for e in batch["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(filter_func, batched=True)
    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(path_to_save)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Translation Data Prep")

    parser.add_argument(
        "--test_split_pct", 
        default=0.005, 
        help="What percent of data do you want to use for Train/Test split",
        type=float
    )

    parser.add_argument(
        "--max_length", 
        default=512, 
        help="Pass in argument to override the default in Config, but then make sure config \
            reflects this when training",
        type=int
    )

    parser.add_argument(
        "--min_length", 
        default=5, 
        help="Removes any token sequences that are shorter than this length",
        type=int
    )

    parser.add_argument(
        "--path_to_data_root", 
        required=True, 
        help="Path to where you want to save the final tokenized dataset",
        type=str
    )

    parser.add_argument(
        "--huggingface_cache_dir",
        default=None,
        help="path to huggingface cache directory if different from default",
        type=str
    )

    parser.add_argument(
        "--num_workers",
        default=24, 
        help="Number of workers you want to use to process dataset",
        type=int
    )

    args = parser.parse_args()

    path_to_data_root = args.path_to_data_root
    path_to_data_raw = os.path.join(path_to_data_root, "raw_english2french_corpus")
    path_to_data_tokenized = os.path.join(path_to_data_root, "tokenized_english2french_corpus")
    cache_dir = args.huggingface_cache_dir

    build_english2french_dataset(path_to_data_root, 
                                 path_to_data_raw, 
                                 test_prop=args.test_split_pct, 
                                 cache_dir=cache_dir)

    tokenize_english2french_dataset(path_to_data_raw, 
                                    path_to_data_tokenized, 
                                    truncate=True,
                                    max_length=args.max_length,
                                    min_length=args.min_length, 
                                    num_workers=args.num_workers)