import json
import transformers
import torch
from tqdm import tqdm
import os
from utils.prompt_structure import (
    BasePrompt,
    DescPrompt,
    Head2SpanPrompt,
)


def load_few_shots(few_shot_file):
    few_shot_instances = []
    with open(few_shot_file) as read_f:
        for line in read_f:
            instance = json.loads(line.strip())
            few_shot_instances.append(instance)

    return few_shot_instances


def get_kwargs(args):
    model_kwargs = {"trust_remote_code": True}

    if args.precision == 16:
        if transformers.utils.is_torch_bf16_gpu_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

    elif args.precision == 8:
        model_kwargs["load_in_8bit"] = True

    tokenizer_kwargs = {"trust_remote_code": True}
    return model_kwargs, tokenizer_kwargs


def extract_cluster_names(key_entities):
    cluster_names = []
    for key_entity in key_entities:
        start_idx = key_entity.index("(")
        end_idx = key_entity.index(")")

        cluster_name = key_entity[start_idx + 1 : end_idx]
        cluster_names.append(cluster_name)

    return cluster_names


def check_all_output_exists(args):
    instances = [
        json.loads(line.strip()) for line in open(args.paths.eval_file).readlines()
    ]
    for idx, test_instance in tqdm(enumerate(instances), total=len(instances)):
        output_file = args.paths.output_folder / f"{test_instance['doc_key']}.txt"
        if not os.path.exists(output_file):
            return False
    return True


def format_key_entities(instance):
    if "key_entities" in instance:
        key_entities = instance["key_entities"]
        key_entities_fmtd = []
        for idx, entity in enumerate(key_entities):
            key_entities_fmtd.append(f"{idx + 1}. {entity} ")
        instance["key_entities_list"] = instance["key_entities"]
        instance["key_entities"] = "\n".join(key_entities_fmtd)
    return instance


def create_prompt(instruction, few_shot_instances, cot, h2s=False):
    # Create prompt
    if not h2s:
        if cot == "desc":
            prompt = DescPrompt(instruction=instruction, examples=few_shot_instances)
        else:
            prompt = BasePrompt(instruction=instruction, examples=few_shot_instances)
    else:
        prompt = Head2SpanPrompt(instruction=instruction, examples=few_shot_instances)
    return prompt


def get_inst_fewshots(instruction_file, few_shot_file):
    # Load instruction and few shot instances
    instruction = open(instruction_file).read()
    few_shot_instances = []
    for instance in load_few_shots(few_shot_file=few_shot_file):
        instance = format_key_entities(instance)
        few_shot_instances.append(instance)
    return instruction, few_shot_instances
