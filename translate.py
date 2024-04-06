# ----- IMPORTS ----- #

import json
import asyncio
import random
import time

from models.openai.openai_models import complete, complete_all 
from models.openai.format_prompt import format_prompt as openai_format_prompt

# ----- HELPER FUNCTIONS ----- #

def __default_format(prompt):  return prompt

def __generate_random_seed():
    return str(random.randint(10000, 100000)) + '\n\n'

def __extract_list_from_string(s: str):
    start = s.find("[")
    end = s.find("]") + 1
    if start == -1 or end == 0:
        return []  # Return an empty list if no list is found
    lst_str = s[start:end]
    try:
        return json.loads(lst_str)
    except:
        return []


def __get_format_prompt_function(model):
    match model:
        case "gpt-4-0125-preview":
            return openai_format_prompt
        case "gpt-3.5-turbo":
            return openai_format_prompt
        case _:
            return __default_format

# ----- MAIN FUNCTIONS ----- #

async def translate(model, prompt, input_filepath, output_filepath, batch_size = 20):

    start_time = time.time()  # Start timing

    format_prompt = __get_format_prompt_function(model)

    with open(input_filepath, 'r') as file:
        dataset = json.load(file)

    prompt_prompts = [format_prompt(prompt + data["prompt"]) for data in dataset]
    completion_prompts = [format_prompt(prompt + data["completion"]) for data in dataset]
    
    translated_prompts = await complete_all(
        batch_size,
        prompt_prompts,
        model
    )
    translated_completions = await complete_all(
        batch_size,
        completion_prompts,
        model
    )

    translated_dataset = []

    for i, translation in enumerate(translated_prompts):
        if (translation and translated_completions[i]):
            translated_dataset.append({
                "prompt": translation,
                "completion": translated_completions[i]
            })

    print(translated_dataset)

    with open(output_filepath, 'w') as file:
        json.dump(translated_dataset, file)

    end_time = time.time()  # End timing
    print(f"Dataset translation completed in {end_time - start_time} seconds.")


if __name__ == '__main__':
    MODEL = 'gpt-4-0125-preview'
    PROMPT = "Translate the following into Modern Greek. Don't give notes, just give a translation:"
    INPUT_FILEPATH = 'data/instruct_no_notes.json'
    OUTPUT_FILEPATH = 'data/translated/el_no_notes.json'
    asyncio.run(translate(MODEL, PROMPT, INPUT_FILEPATH, OUTPUT_FILEPATH))