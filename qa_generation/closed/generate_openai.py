import os
import sys
import jsonlines
import wandb
import hydra

from openai import OpenAI
from dotenv import load_dotenv

from tqdm import tqdm
from transformers.utils import logging
from pathlib import Path

from .config import TEMPERATURE, SYSTEM_MESSAGE
from configs.config_gen import NAME_TO_PREFIX

from utils.qa_utils import check_all_outputs_exists, get_inst_format

from qa_generation.evaluate import evaluate
from utils.prompt_structure import QAPrompt


logger = logging.get_logger("COREFERENCE")


def main(args):
    print(args.model_name)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        logger.warning(f"Index: {idx}")
        test_instance_key = (test_instance["doc_key"], test_instance["mention_ind"])

        if test_instance_key in existent_outputs:
            logger.warning(f"Output for {test_instance['doc_key']} exists. Skipping!")
            continue

        prompt_string = prompt.populate_prompt(test_instance)

        if not args.no_wandb:
            wandb.log({"prompt": prompt_string})

        output_string = ""
        conversation = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_string},
        ]

        logger.warning("Generate the output")
        try:
            completion = client.chat.completions.create(
                model=args.model_name,
                messages=conversation,
                temperature=TEMPERATURE,
                max_tokens=4095,
                stream=True,
            )

            for chunk in completion:
                if chunk.choices[0].delta.content != None:
                    output_string += chunk.choices[0].delta.content
                else:
                    finish_reason = chunk.choices[0].finish_reason

        except Exception as e:
            print("Error in Document ID:", test_instance["doc_key"])
            print(conversation)
            logger.error(f"Error: {e}")
            print("Error: ", e)
            continue

        if finish_reason == "content_filter":
            print("Content Filter Error in Document ID:", test_instance["doc_key"])
            logger.error(f"Content Filter Error: {completion}")
            continue

        if output_string != "":
            if not args.no_wandb:
                wandb.log({"output": output_string})

            predicted_answer = output_string.split("The mention refers to: ")[-1].strip()

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
    print("Model Prefix: ", args.model_prefix)
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
                "setup": args.setup,
                "cot": args.cot,
                "precision": args.precision,
            },
        )

    load_dotenv()  # take environment variables from .env
    main(args)


if __name__ == "__main__":
    init()
