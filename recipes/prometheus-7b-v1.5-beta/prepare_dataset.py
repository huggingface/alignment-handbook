from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def prepare_dataset():
    cache_dir = "/mnt/sda/juyoung/cache"
    dataset_1 = load_dataset('kaist-ai/Feedback-Collection', cache_dir=cache_dir)
    dataset_2 = load_dataset('kaist-ai/Preference-Collection', cache_dir=cache_dir)

    df_1 = dataset_1['train'].to_pandas()
    df_2 = dataset_2['train'].to_pandas()

    system_prompt = "You are a fair judge assistant responsible for writing an incisive and insightful feedback."

    def add_messages_column(row):
        system_msg = {"content": system_prompt, "role": "system"}
        user_msg = {"content": row["instruction"], "role": "user"}
        assistant_msg = {"content": row["output"], "role": "assistant"}
        messages = [system_msg, user_msg, assistant_msg]
        row['messages'] = messages
        return row

    df_1 = df_1.apply(add_messages_column, axis=1)
    df_2 = df_2.apply(add_messages_column, axis=1)
    
    Path('./assets/feedback-collection').mkdir(parents=True, exist_ok=True)
    Path('./assets/preference-collection').mkdir(parents=True, exist_ok=True)
    
    dataset_1 = Dataset.from_pandas(df_1)
    dataset_1 = dataset_1.save_to_disk('./assets/feedback-collection')
    
    dataset_2 = Dataset.from_pandas(df_2)
    dataset_2 = dataset_2.save_to_disk('./assets/preference-collection')


if __name__ == "__main__":
    prepare_dataset()
    dataset_1 = load_from_disk('./assets/feedback-collection')
    print("Hello, World!")
    # dataset_2 = load_dataset('./assets/preference-collection')
    # import pdb; pdb.set_trace()