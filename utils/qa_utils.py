import os
import random
import jsonlines
import copy
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
from thefuzz import fuzz
from configs.config import PRONOUNS_GROUPS, PLURAL_PRONOUNS, selected_keys


def write_qa_to_jsonl(df, dataset_proc, major_entities, dataset_jsonl):
    ## Create directory if not present
    os.makedirs(os.path.dirname(dataset_jsonl), exist_ok=True)
    data_list = []
    noa_indices = np.random.choice(
        np.arange(len(df)), int(0.1 * len(df)), replace=False
    )
    for row_index, row in enumerate(df.iterrows()):
        doc_key = row[1]["doc_key"]
        mention_tbound = row[1]["mentions_vs_tbound"]

        tokens_doc = copy.deepcopy(dataset_proc[doc_key]["token_vs_tokenstr"])
        tokens_doc[mention_tbound[0]] = f"{{{{{tokens_doc[mention_tbound[0]]}"
        tokens_doc[mention_tbound[1]] = (
            f"{tokens_doc[mention_tbound[1]]}}}}} (#This is the marked mention)"
        )

        text = " ".join(tokens_doc)

        entity_name_options = copy.deepcopy(major_entities[doc_key]["entity_name"])
        random.shuffle(entity_name_options)

        options = entity_name_options + ["None of the Above"]
        answer = row[1]["entity_name"]
        orig_answer = row[1]["entity_name"]

        if row_index in noa_indices:
            options.remove(answer)
            answer = "None of the Above"

        data_list.append(
            {
                "doc_key": doc_key,
                "mention_ind": row[1]["mention_ind"],
                "category": row[1]["category"],
                "text": text,
                "options": options,
                "answer": answer,
                "orig_answer": orig_answer,
            }
        )

    ## Write to jsonl
    with jsonlines.open(dataset_jsonl, mode="a") as writer:
        writer.write_all(data_list)


def check_all_outputs_exists(args):
    with jsonlines.open(args.paths.eval_file) as reader:
        instances = {
            (instance["doc_key"], instance["mention_ind"]) for instance in reader
        }

    if not os.path.exists(args.paths.output_file):
        return False
    else:
        with jsonlines.open(args.paths.output_file) as reader:
            outputs = {(output["doc_key"], output["mention_ind"]) for output in reader}

        if outputs == instances:
            return True
        else:
            return False


def get_inst_format(instruction_file, format_file):
    # Load instruction and few shot instances
    instruction = open(instruction_file).read()
    format = open(format_file).read()
    return instruction, format


def get_rejected_docs(dataset_proc, major_entities):
    rejected_doc_list = []
    for doc in dataset_proc:
        entity_id = major_entities[doc]["entity_id"]
        total_mentions = len(dataset_proc[doc]["mentions_vs_stbound"])
        mentions_annotated = len(
            [
                1
                for cluster in dataset_proc[doc]["mentions_vs_clusters"]
                if cluster in entity_id
            ]
        )
        mentions_lost = total_mentions - mentions_annotated
        percent_lost = (mentions_lost / total_mentions) * 100
        num_tokens = len(dataset_proc[doc]["token_vs_tokenstr"])
        if num_tokens < 1000 or percent_lost > 50:
            rejected_doc_list.append(doc)
    return rejected_doc_list


def get_pairwise_rank(column_det, copelands_count):
    col_pairwise = [0 for i in range(len(column_det))]
    for i in tqdm(range(len(column_det))):
        for j in range(i + 1, len(column_det)):
            if column_det[i] == column_det[j]:
                col_pairwise[i] += 0.5
                col_pairwise[j] += 0.5
            elif column_det[i] > column_det[j]:
                col_pairwise[i] += 1
            else:
                col_pairwise[j] += 1
    for i in range(len(col_pairwise)):
        copelands_count[i] += col_pairwise[i]
    arg_sorted_lst = np.argsort(col_pairwise).tolist()
    col_ranks = [
        (len(col_pairwise)) - arg_sorted_lst.index(i) for i in range(len(col_pairwise))
    ]
    return copelands_count, col_ranks


def add_copelands_count(mention_info_df, type="nom"):
    ## Add copelands_ranking column to mention_info_df
    copelands_count = [0 for i in range(len(mention_info_df))]
    if type == "nom":
        copelands_count, col_rank = get_pairwise_rank(
            (-np.array(mention_info_df["name_fuzzy_scores"])).tolist(), copelands_count
        )
        mention_info_df["name_fuzzy_scores_rank"] = col_rank
    else:
        copelands_count, col_rank = get_pairwise_rank(
            list(mention_info_df["fin_dists"]), copelands_count
        )
        mention_info_df["fin_dists_rank"] = col_rank

    copelands_count, col_rank = get_pairwise_rank(
        list(mention_info_df["neighbour_distance"]), copelands_count
    )
    mention_info_df["neighbour_distance_rank"] = col_rank

    copelands_count, col_rank = get_pairwise_rank(
        list(mention_info_df["nominal_neighbour_distance"]), copelands_count
    )
    mention_info_df["nominal_neighbour_distance_rank"] = col_rank

    copelands_count, col_rank = get_pairwise_rank(
        list(mention_info_df["surface_neighbour_distance"]), copelands_count
    )
    mention_info_df["surface_neighbour_distance_rank"] = col_rank

    mention_info_df["copelands_count"] = copelands_count

    arg_sorted_lst = np.argsort(copelands_count).tolist()
    mention_info_df["copelands_ranking"] = [
        len(mention_info_df) - arg_sorted_lst.index(i)
        for i in range(len(mention_info_df))
    ]
    return mention_info_df


def get_mention_info(dataset_proc, major_entities, tsv_addr=None, rejected_docs=[]):
    mention_info_dict = defaultdict(list)
    MAX_NUM = 9999999
    for document in dataset_proc:
        if document in rejected_docs:
            continue
        major_entity_names = major_entities[document]["entity_name"]
        major_entity_ids = major_entities[document]["entity_id"]

        mention_info_dict["doc_key"].extend(
            [document] * len(dataset_proc[document]["mentions_vs_stbound"])
        )
        mention_info_dict["mention_ind"].extend(
            list(range(len(dataset_proc[document]["mentions_vs_stbound"])))
        )

        for key in selected_keys:
            mention_info_dict[key].extend(dataset_proc[document][key])

        pronoun_class_list = []
        category_list = []
        if tsv_addr is not None:
            category_list = dataset_proc[document]["mentions_vs_mentionctgry"]
            for token in dataset_proc[document]["mentions_vs_mentionstr"]:
                if token.lower().strip() in PRONOUNS_GROUPS:
                    pronoun_class_list.append(PRONOUNS_GROUPS[token.lower().strip()])
                else:
                    pronoun_class_list.append(-1)
        else:
            for token in dataset_proc[document]["mentions_vs_mentionstr"]:
                if token.lower().strip() in PRONOUNS_GROUPS:
                    category_list.append("PRON")
                    pronoun_class_list.append(PRONOUNS_GROUPS[token.lower().strip()])
                else:
                    category_list.append("NOM")
                    pronoun_class_list.append(-1)
        mention_info_dict["category"].extend(category_list)
        mention_info_dict["pronoun_class"].extend(pronoun_class_list)

        entity_cluster_list = dataset_proc[document]["mentions_vs_clusters"]
        entity_names_list = []
        for cluster in entity_cluster_list:
            if cluster in major_entity_ids:
                entity_names_list.append(
                    major_entity_names[major_entity_ids.index(cluster)]
                )
            else:
                entity_names_list.append("Others")
        mention_info_dict["entity_name"].extend(entity_names_list)

        name_fuzzy_scores_list = []
        for mention_ind, cluster in enumerate(
            dataset_proc[document]["mentions_vs_clusters"]
        ):
            if cluster in major_entity_ids:
                entity_name = entity_names_list[mention_ind]
                mention_str = dataset_proc[document]["mentions_vs_mentionstr"][
                    mention_ind
                ]
                name_fuzzy_scores_list.append(
                    fuzz.partial_token_sort_ratio(
                        entity_name.lower(), mention_str.lower()
                    )
                )
            else:
                name_fuzzy_scores_list.append(-1)
        mention_info_dict["name_fuzzy_scores"].extend(name_fuzzy_scores_list)

        neighbour_distance_list = []
        nominal_neighbour_distance_list = []
        surface_neighbour_distance_list = []
        for mention_ind, cluster in enumerate(
            dataset_proc[document]["mentions_vs_clusters"]
        ):
            cluster_mentions = np.array(
                dataset_proc[document]["clusters_vs_mentions"][cluster]
            )
            nominal_mentions = np.array(
                [i for i in cluster_mentions if category_list[i] != "PRON"]
            )
            surface_mentions = np.array(
                [
                    i
                    for i in cluster_mentions
                    if category_list[i] != "PRON" and name_fuzzy_scores_list[i] > 75
                ]
            )

            cluster_distances = np.abs(cluster_mentions - mention_ind)
            cluster_distances = np.delete(
                cluster_distances, np.where(cluster_distances == 0)
            )
            if len(cluster_distances) > 0:
                neighbour_distance_list.append(np.min(cluster_distances))
            else:
                neighbour_distance_list.append(-1)

            nominal_distances = np.abs(nominal_mentions - mention_ind)
            nominal_distances = np.delete(
                nominal_distances, np.where(nominal_distances == 0)
            )
            if len(nominal_distances) > 0:
                nominal_neighbour_distance_list.append(np.min(nominal_distances))
            else:
                nominal_neighbour_distance_list.append(-1)

            surface_distances = np.abs(surface_mentions - mention_ind)
            surface_distances = np.delete(
                surface_distances, np.where(surface_distances == 0)
            )
            if len(surface_distances) > 0:
                surface_neighbour_distance_list.append(np.min(surface_distances))
            else:
                surface_neighbour_distance_list.append(-1)

        mention_info_dict["neighbour_distance"].extend(neighbour_distance_list)
        mention_info_dict["nominal_neighbour_distance"].extend(
            nominal_neighbour_distance_list
        )
        mention_info_dict["surface_neighbour_distance"].extend(
            surface_neighbour_distance_list
        )

        avg_fuzzy_scores_list = []
        for mention_ind, cluster in enumerate(
            dataset_proc[document]["mentions_vs_clusters"]
        ):
            ## Not assigning scores for PRONs for now
            if category_list[mention_ind] == "PRON":
                avg_fuzzy_scores_list.append(-1)
                continue
            ## For nominals
            mention_str = dataset_proc[document]["mentions_vs_mentionstr"][mention_ind]
            cluster_mentions = dataset_proc[document]["clusters_vs_mentions"][cluster]
            fuzzy_scores = []
            for cluster_mention in cluster_mentions:
                ## Only calculating for nominals
                if category_list[cluster_mention] != "PRON":
                    fuzzy_scores.append(
                        fuzz.partial_token_sort_ratio(
                            mention_str.lower(),
                            dataset_proc[document]["mentions_vs_mentionstr"][
                                cluster_mention
                            ].lower(),
                        )
                    )

            avg_fuzz_score = (
                sum(fuzzy_scores) / len(fuzzy_scores) if len(fuzzy_scores) > 0 else -1
            )
            avg_fuzzy_scores_list.append(avg_fuzz_score)
        mention_info_dict["avg_fuzzy_scores"].extend(avg_fuzzy_scores_list)

        cluster_size_list = []
        for cluster in dataset_proc[document]["mentions_vs_clusters"]:
            cluster_size_list.append(
                len(dataset_proc[document]["clusters_vs_mentions"][cluster])
            )
        mention_info_dict["cluster_size"].extend(cluster_size_list)

        nominal_size_list = []
        for cluster in dataset_proc[document]["mentions_vs_clusters"]:
            nominal_size_list.append(
                sum(
                    [
                        category_list[i] != "PRON"
                        for i in dataset_proc[document]["clusters_vs_mentions"][cluster]
                    ]
                )
            )
        mention_info_dict["nominal_size"].extend(nominal_size_list)

        is_plural_list = []

        for mention_ind, cluster in enumerate(
            dataset_proc[document]["mentions_vs_clusters"]
        ):
            mentions_str_cluster = [
                dataset_proc[document]["mentions_vs_mentionstr"][i]
                for i in dataset_proc[document]["clusters_vs_mentions"][cluster]
            ]
            is_plural = any(
                [
                    mention_str.lower() in PLURAL_PRONOUNS
                    for mention_str in mentions_str_cluster
                ]
            )
            # print(is_plural)
            is_plural_list.append(is_plural)
        mention_info_dict["is_plural"].extend(is_plural_list)

        ## Number of other major mentions beside (nearest 10 mentions on either side) mention
        is_major = [
            entity_names_list[i] != "Others" for i in range(len(entity_names_list))
        ]
        num_major_mentions = []
        for i in range(len(is_major)):
            num_major_mentions.append(
                sum(is_major[max(0, i - 10) : min(len(is_major), i + 11)])
            )
        mention_info_dict["num_major_mentions_neigh"].extend(num_major_mentions)

        ## Number of neighbours that have same pronoun category and entity name
        same_pn_same_ent = []
        for i in range(len(pronoun_class_list)):
            if pronoun_class_list[i] == -1:
                same_pn_same_ent.append(-1)
            else:
                same_pn_same_ent.append(
                    sum(
                        [
                            1
                            for j in range(
                                max(0, i - 10), min(len(pronoun_class_list), i + 11)
                            )
                            if pronoun_class_list[j] == pronoun_class_list[i]
                            and entity_cluster_list[j] == entity_cluster_list[i]
                        ]
                    )
                )
        mention_info_dict["num_helpers"].extend(same_pn_same_ent)

        ## Number of neighbours that have same pronoun category and different entity name
        same_pn_diff_ent = []
        diff_pn_same_ent = []
        for i in range(len(pronoun_class_list)):
            if pronoun_class_list[i] == -1:
                same_pn_diff_ent.append(-1)
                diff_pn_same_ent.append(-1)
            else:
                same_pn_diff_ent.append(
                    sum(
                        [
                            1
                            for j in range(
                                max(0, i - 10), min(len(pronoun_class_list), i + 11)
                            )
                            if pronoun_class_list[j] == pronoun_class_list[i]
                            and entity_cluster_list[j] != entity_cluster_list[i]
                        ]
                    )
                )
                diff_pn_same_ent.append(
                    sum(
                        [
                            1
                            for j in range(
                                max(0, i - 10), min(len(pronoun_class_list), i + 11)
                            )
                            if pronoun_class_list[j] != -1
                            and pronoun_class_list[j] != pronoun_class_list[i]
                            and entity_cluster_list[j] == entity_cluster_list[i]
                        ]
                    )
                )
        mention_info_dict["num_dists"].extend(
            [harm1 + harm2 for harm1, harm2 in zip(same_pn_diff_ent, diff_pn_same_ent)]
        )

        ## Final distractors
        mention_info_dict["fin_dists"].extend(
            [
                (
                    same_pn_diff_ent[i] + diff_pn_same_ent[i] - same_pn_same_ent[i]
                    if same_pn_same_ent[i] != -1
                    else -1
                )
                for i in range(len(same_pn_diff_ent))
            ]
        )

        ## Determine if the mention is a nested mention and the entities involved in the nest
        mentions_tbound = copy.deepcopy(dataset_proc[document]["mentions_vs_tbound"])
        ## Sort by ascending order of start index and then descending order of end index and map to original indices
        mentions_tbound = sorted(mentions_tbound, key=lambda x: (x[0], -x[1]))
        mentions_tbound_orig_map = [
            dataset_proc[document]["mentions_vs_tbound"].index(mention)
            for mention in mentions_tbound
        ]

        ## Create a dictionary of mentions that contains other mentions
        mention_contains_dict = {}
        for i in range(len(mentions_tbound)):
            mention_contains_dict[mentions_tbound_orig_map[i]] = []
            for j in range(i + 1, len(mentions_tbound)):
                if (
                    mentions_tbound[j][0] >= mentions_tbound[i][0]
                    and mentions_tbound[j][1] <= mentions_tbound[i][1]
                ):
                    mention_contains_dict[mentions_tbound_orig_map[i]].append(
                        mentions_tbound_orig_map[j]
                    )
                elif mentions_tbound[j][0] > mentions_tbound[i][1]:
                    break

        ## Create a dictionary that is contained by other mentions
        mention_contained_dict = {}
        for mention in mention_contains_dict:
            for contained_mention in mention_contains_dict[mention]:
                if contained_mention not in mention_contained_dict:
                    mention_contained_dict[contained_mention] = []
                mention_contained_dict[contained_mention].append(mention)

        ## Determine if the mention is a nested mention
        is_nested = []
        for mention_ind in range(len(mentions_tbound_orig_map)):
            if (
                len(mention_contains_dict.get(mention_ind, [])) > 0
                or len(mention_contained_dict.get(mention_ind, [])) > 0
            ):
                is_nested.append(1)
            else:
                is_nested.append(0)

        mention_info_dict["is_nested"].extend(is_nested)

        ## Entities involved in the nest
        nested_entities = []
        for mention_ind in range(len(mentions_tbound_orig_map)):
            entities_involved = set()
            for nested_mention in mention_contains_dict.get(mention_ind, []):
                entities_involved.add(entity_names_list[nested_mention])
            for nested_mention in mention_contained_dict.get(mention_ind, []):
                entities_involved.add(entity_names_list[nested_mention])
            nested_entities.append(list(entities_involved))

        mention_info_dict["nested_entities"].extend(nested_entities)

        ## Need to add pronoun class. For pronouns it is the same. Each entity is assigned with the maximum pronoun class
        

    for key in mention_info_dict:
        print(key, len(mention_info_dict[key]))
    mention_info_df = pd.DataFrame(mention_info_dict)

    return mention_info_df
