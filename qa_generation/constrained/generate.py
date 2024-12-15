import os
import sys
import jsonlines
import wandb
import hydra
import re
from time import sleep

import outlines
from outlines.samplers import GreedySampler

from tqdm import tqdm
from transformers.utils import logging
from pathlib import Path

from configs.config_gen import NAME_TO_PREFIX
from utils.generate_utils import get_kwargs
from utils.qa_utils import check_all_outputs_exists, get_inst_format
from qa_generation.evaluate import evaluate
from utils.prompt_structure import QAPrompt

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = logging.get_logger("COREFERENCE")


def create_decoding_regex(options, args, category):

    regex_opts = "(" + "|".join(options) + ")"
    if args.cot == "desc":
        expl_regex = rf"- Explanation: [A-Za-z ,\'\.]{{150,350}}\.\n"
    else:
        expl_regex = ""

    decoding_regex = rf"- Mention: \{{\{{[A-Za-z ,\'\.]{{1,125}}\}}\}}\n{expl_regex}- The mention refers to: {regex_opts}"

    print("Decoding Regex: ", decoding_regex)
    return decoding_regex


def main(args):
    print(args.model_name)
    model_kwargs, tokenizer_kwargs = get_kwargs(args)

    model = outlines.models.transformers(
        args.model_name,
        device=args.device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    sampler = GreedySampler()

    instruction, format = get_inst_format(
        args.paths.instruction_file, args.paths.format_file
    )

    prompt = QAPrompt(instruction, format)

    with jsonlines.open(args.paths.eval_file) as reader:
        instances = [instance for instance in reader]

    output_file = args.paths.output_file
    existent_outputs = []

    if os.path.exists(output_file):
        logger.warning(f"Outputs partially exist. Skipping!")
        existent_outputs = [
            (output["doc_key"], output["mention_ind"])
            for output in jsonlines.open(output_file)
        ]
    else:
        if not os.path.exists(args.paths.output_file.parent):
            os.makedirs(args.paths.output_file.parent)

    for idx, test_instance in tqdm(enumerate(instances), total=len(instances)):
        test_instance_key = (test_instance["doc_key"], test_instance["mention_ind"])

        if test_instance_key in existent_outputs:
            logger.warning(f"Output for {test_instance['doc_key']} exists. Skipping!")
            continue

        category = test_instance["category"]
        print(f"Category: {category}")
        decoding_regex = create_decoding_regex(test_instance["options"], args, category)
        prompt_string = prompt.populate_prompt(test_instance)

        if not args.no_wandb:
            wandb.log({"prompt": prompt_string})

        output_string = ""

        try:
            # Building the regex processor
            logger.warning("Building the regex processor")
            print("Building the regex processor")
            generator = outlines.generate.regex(model, decoding_regex, sampler=sampler)

            # Generating the output
            print("Generating the output", flush=True)
            logger.warning("Generate the output")
            output_string = generator(prompt_string)

        except Exception as e:
            print("Error in Document ID:", test_instance["doc_key"])
            logger.error(f"Error: {e}")
            print("Error: ", e)

        if output_string != "":
            if not args.no_wandb:
                wandb.log({"output": output_string})

            predicted_answer = output_string.split("The mention refers to: ")[
                -1
            ].strip()

            ## Added code to save the outputs in jsonl file directly
            print("Output produced: ", output_string)
            print("Predicted Answer: ", predicted_answer)
            print("Answer: ", test_instance["answer"])

            with jsonlines.open(output_file, mode="a") as writer:
                writer.write(
                    {
                        "doc_key": test_instance["doc_key"],
                        "mention_ind": test_instance["mention_ind"],
                        "category": test_instance["category"],
                        "text": test_instance["text"],
                        "prompt": prompt_string,
                        "options": test_instance["options"],
                        "answer": test_instance["answer"],
                        "output": output_string,
                        "predicted_answer": predicted_answer,
                    }
                )
    evaluate(args)


@hydra.main(config_path=f"{os.getcwd()}/configs/args", config_name="args_qa")
def init(args):
    args["model_prefix"] = NAME_TO_PREFIX.get(args.model_name, args.model_name[:5])
    if args.precision != 16:
        args["model_prefix"] = f"{args.model_prefix}_precision_{args.precision}"

    for key in args.paths:
        args.paths[key] = Path(args.paths[key])

    if check_all_outputs_exists(args):
        print("All outputs already exist!")
        sys.exit(0)

    if not args.no_wandb:
        wandb.init(
            project="major-entity-identification-qa",
            config={
                "model": args.model_name,
                "dataset": args.dataset,
                "cot": args.cot,
                "precision": args.precision,
            },
        )

    main(args)


if __name__ == "__main__":
    init()
