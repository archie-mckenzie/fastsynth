def format_prompt(prompt):
    new_prompt = {
        "role": "user",
        "content": prompt
    }
    return [new_prompt]