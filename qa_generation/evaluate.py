import os
from pathlib import Path
import sys
import hydra
import jsonlines
from configs.config_gen import NAME_TO_PREFIX

from utils.qa_utils import check_all_outputs_exists


def evaluate(args):
    output_file = args.paths.output_file
    with jsonlines.open(output_file) as reader:
        output_list = list(reader)

    input_format_parts = ["{{", "}} (#This is the marked mention)"]

    if args.cot == "desc":
        output_format_parts = [
            "- Mention: ",
            "- Explanation:",
            "- The mention refers to:",
        ]
    else:
        output_format_parts = ["- Mention: ", "- The mention refers to:"]

    total = len(output_list)
    total_correct = 0
    total_other_correct = 0
    total_phrase_correct = 0
    total_corr_count = 0

    for output_ind, output in enumerate(output_list):
        answer = output["answer"].strip()
        output_text = output["output"]
        text = output["text"]
        predicted_answer = output.get("predicted_answer", None)

        if predicted_answer is None:
            predicted_answer = (
                output_text.split("The mention refers to: ")[-1]
                .replace("*", "")
                .strip()
            )

        phrase = (
            text.split(input_format_parts[1])[0]
            .split(input_format_parts[0])[-1]
            .strip()
        )
        phrase_extr = (
            output_text.split(output_format_parts[0])[-1]
            .strip()
            .split(output_format_parts[1])[0]
            .strip()
            .replace("{", "")
            .replace("}", "")
            .strip()
        )

        correct = predicted_answer == answer
        phrase_correct = phrase_extr == phrase

        # print("Phrase, len(phrase): ", phrase, len(phrase))
        # print("Phrase_extr, len(phrase_extr): ", phrase_extr, len(phrase_extr))
        # print("Why no length phrase extr", len(phrase_extr))
        if not correct:
            print(
                f"Index: {output_ind} Answer: {answer}, Predicted Answer: {predicted_answer}"
            )
            print("Extracted text from the text: ", phrase)
            print("Extracted phrase from output: ", phrase_extr)
            print("Correct: ", correct)
            print("Phrase Correct: ", phrase_correct)
            print("Correlation: ", phrase_correct and correct)

        total_correct += correct
        total_other_correct += (
            answer == "None of the Above" and predicted_answer == answer
        )
        total_phrase_correct += phrase_correct
        total_corr_count += phrase_correct and correct

    ## Build string with the above counts and accuracy and write to result_file
    result_string = f"Total: {total}\nCorrect: {total_correct}\nOther Correct: {total_other_correct}\nPhrase Correct: {total_phrase_correct}\nCorrelation Count: {total_corr_count}\n"
    acc_string = f"Accuracy: {total_correct}/{total} = {total_correct/total}"
    final_string = result_string + acc_string
    print(final_string)
    with open(args.paths.result_file, "w") as write_f:
        write_f.write(final_string)


@hydra.main(config_path=f"{os.getcwd()}/configs/args", config_name="args_qa")
def init(args):

    args["model_prefix"] = NAME_TO_PREFIX.get(args.model_name, args.model_name[:5])
    for key in args.paths:
        args.paths[key] = Path(args.paths[key])

    # if not (check_all_outputs_exists(args)):
    #     print("All outputs do not exist!")
    #     sys.exit(0)

    evaluate(args)


if __name__ == "__main__":
    init()
