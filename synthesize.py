# synthesize.py
# Synthesizes a dataset using a given config.json file
# Author: Archie McKenzie, 2024

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
    return json.loads(lst_str)


def __get_format_prompt_function(model):
    match model:
        case "gpt-4-0125-preview":
            return openai_format_prompt
        case "gpt-3.5-turbo":
            return openai_format_prompt
        case _:
            return __default_format

# ----- MAIN FUNCTIONS ----- #

async def synthesize(config_filepath):

    start_time = time.time()  # Start timing
    
    with open(config_filepath, 'r') as file:
        config = json.load(file)
    
    model = config["model"]
    dataset_size = config["dataset_size"]
    batch_size = config["batch_size"]
    input_prompt = config["input_prompt"]
    output_prompt = config["output_prompt"]
    supplementary_prompts = config["supplementary_prompts"]
    output_filepath = config["output_filepath"]
    use_random_seed = config["use_random_seed"]
    format_prompt = __get_format_prompt_function(config["model"])
    kwargs = config["model_kwargs"]

    dataset = []

    while len(dataset) <= dataset_size:
        
        prompts = []

        while len(prompts) <= dataset_size - len(dataset):

            if use_random_seed: 
                new_input_prompt = f'{__generate_random_seed()}{input_prompt}\n\nGive {batch_size} diverse examples. Write each example as a string, writing the full list as a list of strings [""].' 
            else: 
                new_input_prompt = f'{input_prompt}\n\nGive {batch_size} diverse examples. Write each example as a string, writing the full list as a list of strings [""].'
            
            formatted_prompt = format_prompt(new_input_prompt)

            new_completion = await complete(
                formatted_prompt, model, **kwargs
            )

            new_prompts = __extract_list_from_string(new_completion)

            prompts.extend(new_prompts)

        prompts = prompts[0: dataset_size]
        completion_prompts = [
            format_prompt(prompt + '\n\n' + output_prompt) for prompt in prompts 
        ]

        completions = await complete_all(
            batch_size, 
            completion_prompts,
            model,
            **kwargs
        )

        for i, completion in enumerate(completions):
            if completion:
                if len(supplementary_prompts) > 0:
                    dataset.append({
                        "prompt": prompts[i] + '\n\n' + random.choice(supplementary_prompts),
                        "completion": completion
                    })
                else: 
                    dataset.append({
                        "prompt": prompts[i],
                        "completion": completion
                    })

    print(dataset)

    with open(output_filepath, 'w') as file:
        json.dump(dataset, file)

    end_time = time.time()  # End timing
    print(f"Dataset synthesis completed in {end_time - start_time} seconds.")


def main():
    CONFIG_FILEPATH = 'config/greek_no_notes.json'
    asyncio.run(synthesize(CONFIG_FILEPATH))

if __name__ == '__main__':
    main()