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


class args:
    data_files = ""
    source_model_checkpoint = "t5-base"
    target_model_checkpoint = "sonoisa/t5-base-japanese"
    output_dir = "outputs/en2ja_t5-base"
    cache_dir = os.getenv("HF_DATASETS_CACHE")
    # WECHSEL
    source_lang_code = "en"
    target_lang_code = "ja"
    bilingual_dict = "japanese"


def report(message):
    print(message)
    # logger.info(message)


def main(args):
    if os.path.exists(args.output_dir):
        report(f"Output dir `{args.output_dir}` already exists")
        sys.exit(0)
    else:
        os.makedirs(args.output_dir)
        report(f"Created `{args.output_dir}`")

    # source
    source_tokenizer = T.AutoTokenizer.from_pretrained(args.source_model_checkpoint, cache_dir=args.cache_dir)
    source_model = T.AutoModelForSeq2SeqLM.from_pretrained(args.source_model_checkpoint, cache_dir=args.cache_dir)
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
    main(args)
