"""Prompt structure."""

import outlines


@outlines.prompt
def base_few_shots(instruction, examples, new_query):
    """{{ instruction }}

    {% for example in examples %}
    Input:

    Key Entities:
    {{ example.key_entities }}

    Text:
    {{ example.text }}

    Output:

    {{ example.output }}

    {% endfor %}

    Input:

    Key Entities:
    {{ new_query.key_entities }}

    Text:
    {{ new_query.text }}

    Output:

    """


@outlines.prompt
def desc_few_shots(instruction, examples, new_query):
    """{{ instruction }}

    {% for example in examples %}
    Input:

    Key Entities:
    {{ example.key_entities }}

    Text:
    {{ example.text }}

    Output:

    Description of Key Entities present in the text:
    {{ example.desc }}

    Coreference:
    {{ example.output }}

    {% endfor %}

    Input:

    Key Entities:
    {{ new_query.key_entities }}

    Text:
    {{ new_query.text }}

    Output:

    """


@outlines.prompt
def head2span_few_shots(instruction, examples, new_query):
    """{{ instruction }}

    {% for example in examples %}
    Input:

    {{ example.text }}

    Output:

    {{ example.output }}

    {% endfor %}

    Input:

    {{ new_query.text }}

    Output:

    """


@outlines.prompt
def qa_prompt(instruction, format, new_query):
    """
    Instruction:
    {{ instruction }}

    Text:
    {{ new_query.text }}

    Options for the answer:
    {% for option in new_query.options %}
    {{ option }}
    {% endfor %}

    {{format}}

    """


@outlines.prompt
def name_prompt(instruction, format, new_query):
    """
    Instruction:
    {{ instruction }}

    Follow the below format:
    {{format}}

    Information:
    {% for key, value in new_query.items() %}
    {{ key }}: {{ value[0] }}
    {% endfor %}
    """


class Prompt:
    def __init__(self, instruction, examples=None, format_str=None):
        self.instruction = instruction
        self.examples = examples
        self.format = format_str
        self.show_prompt = True
        self.prompt_func = None

    def populate_prompt(self, new_query):
        "Construct the few-shot prompt with instruction, examples, and the new query"
        prompt_string = self.prompt_func(self.instruction, self.examples, new_query)

        # Show the first prompt
        if self.show_prompt:
            # print(prompt_string)
            self.show_prompt = False

        return prompt_string


class BasePrompt(Prompt):
    """Base prompt class."""

    def __init__(self, instruction, examples):
        super().__init__(instruction, examples)
        self.prompt_func = base_few_shots


class DescPrompt(Prompt):
    """Desc prompt class."""

    def __init__(self, instruction, examples):
        super().__init__(instruction, examples)
        self.prompt_func = desc_few_shots


class Head2SpanPrompt(Prompt):
    """Head to span prompt class."""

    def __init__(self, instruction, examples):
        super().__init__(instruction, examples)
        self.prompt_func = head2span_few_shots


class QAPrompt(Prompt):
    """QA prompt class."""

    def __init__(self, instruction, format_str):
        super().__init__(instruction, examples=None, format_str=format_str)
        self.prompt_func = qa_prompt

    def populate_prompt(self, new_query):
        "Construct the few-shot prompt with instruction, examples, and the new query"
        prompt_string = self.prompt_func(self.instruction, self.format, new_query)

        # Show the first prompt
        if self.show_prompt:
            self.show_prompt = False

        return prompt_string


class NamePrompt(Prompt):
    """QA prompt class."""

    def __init__(self, instruction, format_str):
        super().__init__(instruction, examples=None, format_str=format_str)
        self.prompt_func = name_prompt

    def populate_prompt(self, new_query):
        "Construct the few-shot prompt with instruction, examples, and the new query"
        prompt_string = self.prompt_func(self.instruction, self.format, new_query)

        # Show the first prompt
        if self.show_prompt:
            self.show_prompt = False

        return prompt_string
