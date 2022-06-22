import argparse
import logging
import os
import sys

import torch

import datasets as D
import transformers as T

from wechsel import WECHSEL, load_embeddings
from utils import Timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def report(message):
    print(message)


def main(args):
    if os.path.exists(args.output_dir):
        report(f"Output dir `{args.output_dir}` already exists")
        sys.exit(0)
    else:
        os.makedirs(args.output_dir)
        report(f"Created `{args.output_dir}`")

    # source
    source_tokenizer = T.AutoTokenizer.from_pretrained(args.source_model_checkpoint, cache_dir=args.cache_dir)
    source_model = T.AutoModelForCausalLM.from_pretrained(args.source_model_checkpoint, cache_dir=args.cache_dir)
    report(source_model.num_parameters())

    # target
    if len(args.data_files) > 0:
        report("Creating target_tokenizer")
        target_tokenizer = source_tokenizer.train_new_from_iterator(
            D.load_dataset(
                "json",
                data_files=args.data_files)['train']['text'],
            vocab_size=len(source_tokenizer)
        )
        target_tokenizer.save_pretrained(args.output_dir)
        report(f"Created and saved target_tokenizer at `{args.output_dir}`")
    else:
        with Timer(prefix=f"Saved target tokenizer at {args.output_dir}"):
            target_tokenizer = T.AutoTokenizer.from_pretrained(args.target_model_checkpoint, cache_dir=args.cache_dir)
            '''
            for k, v in source_tokenizer.special_tokens_map.items():
                prev_v = target_tokenizer.special_tokens_map[k]
                target_tokenizer.add_special_tokens({k: v})
                report(f"Update special tokens `{k}` from `{prev_v}` to `{v}`.")
            '''
            target_tokenizer.save_pretrained(args.output_dir)

    with Timer(prefix="WECHSELed"):
        wechsel = WECHSEL(
            load_embeddings(args.source_lang_code),
            load_embeddings(args.target_lang_code),
            bilingual_dictionary=args.bilingual_dict,
        )

    with Timer(prefix="WECHSEL applied"):
        target_embeddings, info = wechsel.apply(
            source_tokenizer,
            target_tokenizer,
            source_model.get_input_embeddings().weight.detach().numpy(),
        )

    with Timer(prefix=f"Saved pretrained models at {args.output_dir}"):
        source_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
        source_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", type=str, default="")
    parser.add_argument("--source_model_checkpoint", type=str, default="gpt2")
    parser.add_argument("--target_model_checkpoint", type=str, default="rinna/japanese-gpt2-small")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_DATASETS_CACHE"))
    # WECHSEL
    parser.add_argument("--source_lang_code", type=str, default="en")
    parser.add_argument("--target_lang_code", type=str, default="ja")
    parser.add_argument("--bilingual_dict", type=str, default="japanese")

    args, _ = parser.parse_known_args()
    main(args)
