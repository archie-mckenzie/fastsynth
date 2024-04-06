import json

def convert_json_to_jsonl(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    with open(output_filepath, 'w', encoding='utf-8') as file:
        for entry in data:
            jsonl_line = json.dumps(entry)
            file.write(f"{jsonl_line}\n")

if __name__ == '__main__':
    INPUT_FILEPATH = 'data/instruct_no_notes.json'
    OUTPUT_FILEPATH = 'data/jsonl/instruct_no_notes.jsonl'
    convert_json_to_jsonl(INPUT_FILEPATH, OUTPUT_FILEPATH)
