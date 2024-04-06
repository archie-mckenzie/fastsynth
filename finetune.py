import replicate
import requests
from dotenv import load_dotenv
import os
load_dotenv()

def upload(filepath):
    print(os.getenv("REPLICATE_API_TOKEN"))
    headers = {"Authorization": f"Bearer {os.getenv("REPLICATE_API_TOKEN")}"}
    response = requests.post("https://dreambooth-api-experimental.replicate.com/v1/upload/data.jsonl", headers=headers).json()
    
    if 'upload_url' not in response:
        raise KeyError("The response does not contain 'upload_url'. Response: " + str(response))
    
    upload_url = response['upload_url']
    
    with open(filepath, 'rb') as f:
        requests.put(upload_url, data=f)
    
    if 'serving_url' not in response:
        raise KeyError("The response does not contain 'serving_url'. Response: " + str(response))
    
    serving_url = response['serving_url']
    return serving_url

def finetune(training_data_url, destination):

    training = replicate.trainings.create(
        version="mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
        input={
            "train_data": training_data_url,
            "num_train_epochs": 3
        },
        destination=destination
    )

    print(training)


if __name__ == '__main__':
    FILEPATH = "data/jsonl/greek_no_notes.jsonl"
    DESTINATION = 'archie-mckenzie/mistral-greek-no-notes'
    training_data_url = upload(FILEPATH)
    finetune(training_data_url, DESTINATION)


