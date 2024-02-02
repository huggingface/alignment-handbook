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
    
    Path('./assets/feedback-collection/train').mkdir(parents=True, exist_ok=True)
    Path('./assets/feedback-collection/test').mkdir(parents=True, exist_ok=True)
    Path('./assets/preference-collection/train').mkdir(parents=True, exist_ok=True)
    Path('./assets/preference-collection/test').mkdir(parents=True, exist_ok=True)
    
    dataset_1_full = Dataset.from_pandas(df_1)
    dataset_1_full.save_to_disk('./assets/feedback-collection/train')

    dataset_2_full = Dataset.from_pandas(df_2)
    dataset_2_full.save_to_disk('./assets/preference-collection/train')
    
    # Create test datasets with one row each
    df_1_test = df_1.iloc[[0]]  # Select the first row for test set
    df_2_test = df_2.iloc[[0]]  # Select the first row for test set

    dataset_1_test = Dataset.from_pandas(df_1_test)
    dataset_1_test.save_to_disk('./assets/feedback-collection/test')

    dataset_2_test = Dataset.from_pandas(df_2_test)
    dataset_2_test.save_to_disk('./assets/preference-collection/test')


if __name__ == "__main__":
    prepare_dataset()
    #ndataset_1 = load_from_disk('./assets/feedback-collection')
    # dataset_2 = load_dataset('./assets/preference-collection')
    # import pdb; pdb.set_trace()