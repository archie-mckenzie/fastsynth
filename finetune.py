import replicate
from dotenv import load_dotenv
load_dotenv()

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
    DESTINATION = 'archie-mckenzie/hellenic-mistral'
    TRAINING_DATA_URL = 'https://raw.githubusercontent.com/archie-mckenzie/fastsynth/main/data/jsonl/translated_el_instruct_no_notes.jsonl'
    finetune(TRAINING_DATA_URL, DESTINATION)


