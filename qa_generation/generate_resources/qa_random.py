import os
import pandas as pd

from utils.get_processed_dataset import get_processed_dataset
from utils.qa_utils import (
    get_rejected_docs,
    get_mention_info,
    add_copelands_count,
    write_qa_to_jsonl,
)
from utils.utils import read_jsonl, get_major_entities
from omegaconf import OmegaConf
from configs.config import dataset_yaml, PLURAL_PRONOUNS
import numpy as np
import random
import sys


def main():

    RANDOM_SEED = 0
    ## Set seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    TOTAL_SIZE_D_T = 150
    NUM_RANDOM_SETS = 10

    for i in range(NUM_RANDOM_SETS):
        qa_random_dest = f"data/qas/data/qas_random{i}.jsonl"
        if os.path.exists(qa_random_dest):
            os.remove(qa_random_dest)

    dataset_configs = OmegaConf.load(dataset_yaml)

    for dataset in ["litbank", "fantasy"]:
        dataset_source = dataset_configs[dataset][f"train_file"]
        tsv_addr = dataset_configs[dataset][f"tsv"]
        doc_me = dataset_configs[dataset][f"train_me"]
        dataset = read_jsonl(dataset_source)
        dataset_proc = get_processed_dataset(dataset, tsv_litbank=tsv_addr)
        major_entities = get_major_entities(doc_me)
        rejected_docs = get_rejected_docs(dataset_proc, major_entities)
        mention_info_df = get_mention_info(
            dataset_proc,
            major_entities,
            tsv_addr=tsv_addr,
            rejected_docs=rejected_docs,
        )
        mention_info_df = mention_info_df[
            mention_info_df["entity_name"].str.lower() != "others"
        ]

        mention_info_df_nom = mention_info_df[mention_info_df["category"] == "NOM"]

        ## Only condition for nominals to ensure uniformity
        mention_info_df_nom = mention_info_df_nom[
            mention_info_df_nom["mentions_vs_mentionstr"].str.len() <= 125
        ]

        mention_info_df_pron = mention_info_df[mention_info_df["category"] == "PRON"]
        ## Remove plural pronouns -- Should not be present in the df anyways
        mention_info_df_pron = mention_info_df_pron[
            ~mention_info_df_pron["mentions_vs_mentionstr"]
            .str.lower()
            .isin(PLURAL_PRONOUNS)
        ]

        ## Shuffle the dataframes
        mention_info_df_nom = mention_info_df_nom.sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        mention_info_df_pron = mention_info_df_pron.sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)

        for i in range(NUM_RANDOM_SETS):
            qa_nom = mention_info_df_nom.iloc[
                i * TOTAL_SIZE_D_T : (i + 1) * TOTAL_SIZE_D_T
            ].copy()
            qa_nom["category"] = "NOM"
            qa_pron = mention_info_df_pron.iloc[
                i * TOTAL_SIZE_D_T : (i + 1) * TOTAL_SIZE_D_T
            ].copy()
            qa_pron["category"] = "PRON"

            qa_random_dest = f"data/qas/data/qas_random{i}.jsonl"

            write_qa_to_jsonl(qa_nom, dataset_proc, major_entities, qa_random_dest)
            write_qa_to_jsonl(qa_pron, dataset_proc, major_entities, qa_random_dest)


if __name__ == "__main__":
    main()
