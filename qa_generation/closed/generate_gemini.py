import os
import sys
import jsonlines
import wandb
import hydra
from time import sleep

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

from tqdm import tqdm
from transformers.utils import logging
from pathlib import Path

from .config import TEMPERATURE, SYSTEM_MESSAGE, GEMINI_ADD
from configs.config_gen import NAME_TO_PREFIX

from utils.qa_utils import check_all_outputs_exists, get_inst_format

from qa_generation.evaluate import evaluate
from utils.prompt_structure import QAPrompt


logger = logging.get_logger("COREFERENCE")


GENERATION_CONFIG = {
    "temperature": TEMPERATURE,
    "top_p": 1,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def main(args):
    print(args.model_name)
    model_loaded = genai.GenerativeModel(
        model_name=args.model_name,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS,
        system_instruction=SYSTEM_MESSAGE + " " + GEMINI_ADD,
    )

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
        conversation = []
        try:
            chat_session = model_loaded.start_chat(history=conversation)
            print("Prompt: ", prompt_string)
            response = chat_session.send_message(prompt_string)
        except Exception as e:
            print("Error in Document ID:", test_instance["doc_key"])
            # print("Response: ", response)
            logger.error(f"Error: {e}")
            print("Error: ", e)
            continue

        if response.candidates[0].finish_reason.name not in ["STOP", "MAX_TOKENS"]:
            print("Content Filter Error in Document ID:", test_instance["doc_key"])
            logger.error(f"Content Filter Error: {response}")
            continue

        output_string = response.text

        if output_string != "":
            if not args.no_wandb:
                wandb.log({"output": output_string})

            predicted_answer = (
                output_string.split("The mention refers to: ")[-1]
                .replace("*", "")
                .strip()
            )

            ## Added code to save the outputs in jsonl file directly
            print("Output produced: ", output_string)
            print("Predicted Answer: ", predicted_answer)
            print("Answer: ", test_instance["answer"])
            print("Options: ", test_instance["options"])

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

        sleep(5)
    # evaluate(args)


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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    main(args)

    


if __name__ == "__main__":
    init()
