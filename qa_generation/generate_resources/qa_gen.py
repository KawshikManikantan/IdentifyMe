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


def main():
    ## Set seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    TOTAL_SIZE_D_T = 450
    RATIO = int(1 / 3 * TOTAL_SIZE_D_T)

    dataset_configs = OmegaConf.load(dataset_yaml)
    qa_test_dest = "data/qas/data/qas_test.jsonl"
    qa_dev_dest = "data/qas/data/qas_dev.jsonl"

    if os.path.exists(qa_test_dest):
        os.remove(qa_test_dest)
    if os.path.exists(qa_dev_dest):
        os.remove(qa_dev_dest)

    max_pron_length = -1
    max_nom_length = -1
    max_pron = ""
    max_nom = ""

    for dataset in ["litbank", "fantasy"]:
        dataset_source = dataset_configs[dataset][f"train_file"]
        tsv_addr = dataset_configs[dataset][f"tsv"]
        doc_me = dataset_configs[dataset][f"train_me"]
        dataset = read_jsonl(dataset_source)
        dataset_proc = get_processed_dataset(dataset, tsv_litbank=tsv_addr)
        major_entities = get_major_entities(doc_me)
        rejected_docs = get_rejected_docs(dataset_proc, major_entities)
        mention_info_df = get_mention_info(
            dataset_proc, major_entities, tsv_addr=tsv_addr, rejected_docs=rejected_docs
        )
        mention_info_df = mention_info_df[
            mention_info_df["entity_name"].str.lower() != "others"
        ]

        ### Nominals
        mention_info_df_nom = mention_info_df[mention_info_df["category"] == "NOM"]

        ## Remove name_fuzzy_scores > 75 and length >= 125
        mention_info_df_nom = mention_info_df_nom[
            mention_info_df_nom["name_fuzzy_scores"] <= 75
        ]
        mention_info_df_nom = mention_info_df_nom[
            mention_info_df_nom["mentions_vs_mentionstr"].str.len() <= 125
        ]

        mention_info_df_nom = mention_info_df_nom.reset_index(drop=True)
        mention_info_df_nom = add_copelands_count(mention_info_df_nom)

        mention_info_df_pron = mention_info_df[mention_info_df["category"] == "PRON"]
        ## Remove plural pronouns -- Should not be present in the df anyways
        mention_info_df_pron = mention_info_df_pron[
            ~mention_info_df_pron["mentions_vs_mentionstr"]
            .str.lower()
            .isin(PLURAL_PRONOUNS)
        ]
        ## Remove dist <= 0
        mention_info_df_pron = mention_info_df_pron[
            mention_info_df_pron["fin_dists"] > 0
        ]
        mention_info_df_pron = mention_info_df_pron.reset_index(drop=True)
        mention_info_df_pron = add_copelands_count(mention_info_df_pron, type="pron")

        ## Sort by copelands_ranking and select top 300
        selected_df_nom = mention_info_df_nom.sort_values(
            by="copelands_count", ascending=False
        ).head(TOTAL_SIZE_D_T)

        selected_df_pron = mention_info_df_pron.sort_values(
            by="copelands_count", ascending=False
        ).head(TOTAL_SIZE_D_T)

        ## Shuffle the dataframes
        selected_df_nom = selected_df_nom.sample(frac=1, random_state=0).reset_index(
            drop=True
        )
        selected_df_pron = selected_df_pron.sample(frac=1, random_state=0).reset_index(
            drop=True
        )

        ## Get max lengths
        max_pron_length = max(
            max_pron_length, selected_df_pron["mentions_vs_mentionstr"].str.len().max()
        )
        ## Find the pronoun with maximum length
        if (
            selected_df_pron["mentions_vs_mentionstr"].str.len().max()
            == max_pron_length
        ):
            max_pron = selected_df_pron.iloc[
                selected_df_pron["mentions_vs_mentionstr"].str.len().idxmax()
            ]["mentions_vs_mentionstr"]

        max_nom_length = max(
            max_nom_length, selected_df_nom["mentions_vs_mentionstr"].str.len().max()
        )
        ## Find the max nom
        if selected_df_nom["mentions_vs_mentionstr"].str.len().max() == max_nom_length:
            max_nom = selected_df_nom.iloc[
                selected_df_nom["mentions_vs_mentionstr"].str.len().idxmax()
            ]["mentions_vs_mentionstr"]

        ## Split into test and dev
        test_df_nom = selected_df_nom.head(TOTAL_SIZE_D_T - RATIO).copy()
        test_df_nom["category"] = "NOM"
        dev_df_nom = selected_df_nom.tail(RATIO).copy()
        dev_df_nom["category"] = "NOM"
        test_df_pron = selected_df_pron.head(TOTAL_SIZE_D_T - RATIO).copy()
        test_df_pron["category"] = "PRON"
        dev_df_pron = selected_df_pron.tail(RATIO).copy()
        dev_df_pron["category"] = "PRON"

        # print("Test Size Nom: ", test_df_nom.shape[0])
        # print("Dev Size Nom: ", dev_df_nom.shape[0])
        # print("Test Size Pron: ", test_df_pron.shape[0])
        # print("Dev Size Pron: ", dev_df_pron.shape[0])

        write_qa_to_jsonl(test_df_nom, dataset_proc, major_entities, qa_test_dest)
        write_qa_to_jsonl(test_df_pron, dataset_proc, major_entities, qa_test_dest)

        write_qa_to_jsonl(
            dev_df_nom,
            dataset_proc,
            major_entities,
            qa_dev_dest,
        )

        write_qa_to_jsonl(
            dev_df_pron,
            dataset_proc,
            major_entities,
            qa_dev_dest,
        )

    print("Max Nom Length: ", max_nom_length)
    print("Max Nom: ", max_nom)
    print("Max Pron Length: ", max_pron_length)
    print("Max Pron: ", max_pron)


if __name__ == "__main__":
    main()
