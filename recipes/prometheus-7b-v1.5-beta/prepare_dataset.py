from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_dataset():
    cache_dir = "/home/seungone_kim/cache"
    dataset_1 = load_dataset("kaist-ai/Feedback-Collection", cache_dir=cache_dir)
    dataset_2 = load_dataset("kaist-ai/Preference-Collection", cache_dir=cache_dir)

    df_1 = dataset_1["train"].to_pandas()
    df_2 = dataset_2["train"].to_pandas()

    abs_system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
    rel_system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def add_messages_column(row, system_prompt: str):
        system_msg = {"content": system_prompt, "role": "system"}
        user_msg = {"content": row["instruction"], "role": "user"}
        assistant_msg = {"content": row["output"], "role": "assistant"}
        messages = [system_msg, user_msg, assistant_msg]
        row["messages"] = messages
        return row

    # Use lambda function to pass the specific system prompt for each DataFrame
    df_1 = df_1.apply(lambda row: add_messages_column(row, abs_system_prompt), axis=1)
    df_2 = df_2.apply(lambda row: add_messages_column(row, rel_system_prompt), axis=1)

    Path("./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/test").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/test").mkdir(
        parents=True, exist_ok=True
    )

    dataset_1_full = Dataset.from_pandas(df_1)
    dataset_1_full.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/train"
    )

    dataset_2_full = Dataset.from_pandas(df_2)
    dataset_2_full.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/train"
    )

    # Create test datasets with one row each
    df_1_test = df_1.iloc[:32]  # Select the first row for test set
    df_2_test = df_2.iloc[:32]  # Select the first row for test set

    dataset_1_test = Dataset.from_pandas(df_1_test)
    dataset_1_test.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/test"
    )

    dataset_2_test = Dataset.from_pandas(df_2_test)
    dataset_2_test.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/test"
    )


def prepare_dataset_properly():
    cache_dir = "/home/seungone_kim/cache"
    dataset_1 = load_dataset("kaist-ai/Feedback-Collection", cache_dir=cache_dir)
    dataset_2 = load_dataset("kaist-ai/Preference-Collection", cache_dir=cache_dir)

    df_1 = dataset_1["train"].to_pandas()
    df_2 = dataset_2["train"].to_pandas()

    abs_system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
    rel_system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def add_messages_column(row, system_prompt: str):
        # system_msg = {"content": system_prompt, "role": "system"}
        user_msg = {"content": system_prompt + row["instruction"], "role": "user"}
        assistant_msg = {"content": row["output"], "role": "assistant"}
        messages = [user_msg, assistant_msg]
        row["messages"] = messages
        return row

    # Use lambda function to pass the specific system prompt for each DataFrame
    df_1 = df_1.apply(lambda row: add_messages_column(row, abs_system_prompt), axis=1)
    df_2 = df_2.apply(lambda row: add_messages_column(row, rel_system_prompt), axis=1)

    Path("./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/test").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path("./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/test").mkdir(
        parents=True, exist_ok=True
    )

    df_1_train, df_1_test = train_test_split(df_1, test_size=0.01, random_state=42)
    df_2_train, df_2_test = train_test_split(df_2, test_size=0.01, random_state=42)

    dataset_1_train = Dataset.from_pandas(df_1_train)
    dataset_1_train.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/train"
    )

    dataset_2_train = Dataset.from_pandas(df_2_train)
    dataset_2_train.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/train"
    )

    dataset_1_test = Dataset.from_pandas(df_1_test)
    dataset_1_test.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/test"
    )

    dataset_2_test = Dataset.from_pandas(df_2_test)
    dataset_2_test.save_to_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/preference-collection/test"
    )


if __name__ == "__main__":
    prepare_dataset_properly()
    # dataset_1 = load_from_disk('./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/train')
    dataset_2 = load_from_disk(
        "./recipes/prometheus-7b-v1.5-beta/assets/feedback-collection/test"
    )
    print(len(dataset_2))
