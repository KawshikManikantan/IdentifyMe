## Environment
Setup the environment using `environment.yaml`

## Data
Data is uploaded to [huggingface](https://huggingface.co/datasets/KawshikManikantan/IdentifyMe). Do not extract from zip and upload anywhere to prevent data leakage.

## Folder Structure

### `configs/`
- `args/`
  - `args_qa`: Main args file for the experiments
- `config_gen/`
  - Contains model names and prefixes
- `config/`
  - Contains other CONSTANTS

### `data/`
- `human_data`
  - Download from this [gdrive](https://drive.google.com/drive/folders/1uQWT85fJc549asXYKkXSyNX0GWpXSWVv?usp=sharing) for human evaluation data
- `metadata`
  - `gender_info_final.csv`: Contains the gender information of the questions of the benchmark.
  - `pron_dialogue_info.csv`: Contains whether the pronoun is part of a dialogue or not.
- `qas/`
  - `data/`: Download from [gdrive](https://drive.google.com/drive/folders/1_Glr9JJGCFF5Q8nSLet6Ja5TKc9z4Kys?usp=sharing) for all splits. Test split - Benchmark is available at [huggingface]()
    - `qas_dev`: 600-question dev set
    - `qas_test`: 1200-question test set
    - `qas_random_{i}`: 600-question non-repetitive questions
  - `prompts/`
    - `base/`: Prompt information for the version without explanations
    - `desc/`: Prompt information for the version with explanations
    - `coref/`: Prompt information for the version with explanations directed towards coref-chains.
- `raw_data/`: Not necessary for the evaluation alone. Download for reference from [gdrive](https://drive.google.com/drive/folders/1vaVwHhMaDDXLw0rkLzTm5-AOSBDJVNJF?usp=sharing) here

## Commands

### For Constrained Decoding:
```bash
python -m qa_generation.constrained.generate \
    model_name={model_full_name} \
    device="cuda:0/auto" \
    split="dev/test/random_{i}" \
    cot="base/desc/coref"
```

### For GPT Models:
```bash
python -m qa_generation.closed.generate_openai \
    model_name={model_full_name} \
    split="dev/test/random_{i}" \
    cot="base/desc/coref"
```

### For Gemini Models:
```bash
python -m qa_generation.closed.generate_gemini \
    model_name={model_full_name} \
    split="dev/test/random_{i}" \
    cot="base/desc/coref"
```
