# openai_models.py

# ----- IMPORTS ---- #
 
from dotenv import load_dotenv
import os
load_dotenv()

import asyncio
from openai import AsyncOpenAI
client = AsyncOpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

# ----- FUNCTIONS ---- #

async def complete(prompt, model, **kwargs):
    completion = await client.chat.completions.create(
        model=model,
        messages=prompt,
        **kwargs
    )
    return completion.choices[0].message.content


async def complete_all(batch_size, prompts, model = 'gpt-4-0125-preview', **kwargs):
    
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tasks = [complete(prompt, model, **kwargs) for prompt in batch]
        new_results = await asyncio.gather(*tasks)
        results.extend(new_results)

    return results