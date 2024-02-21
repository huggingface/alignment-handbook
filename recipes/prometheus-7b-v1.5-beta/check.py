from transformers import AutoTokenizer


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


import pickle


# Specify the same filename you used for saving
pickle_filename = "/home/seungone_kim/alignment-handbook/batch_labels_1.pkl"

# Open the file in binary-read mode and use pickle.load() to load the object
with open(pickle_filename, "rb") as file:
    loaded_labels = pickle.load(file)

print("Loaded labels from the pickle file:")
print(loaded_labels)


def decode_labels_with_ignore_index(labels, tokenizer, ignore_index=-100):
    decoded_texts = []
    for label_ids in labels:
        # Convert label_ids tensor to list if it is not already
        label_ids = label_ids.tolist()
        # Filter out the ignore_index before decoding
        valid_label_ids = [
            token_id for token_id in label_ids if token_id != ignore_index
        ]
        # Decode the remaining valid token IDs to text
        decoded_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    return decoded_texts


# Assuming batch['labels'] is available after collation
decoded_labels = decode_labels_with_ignore_index(
    loaded_labels, tokenizer, ignore_index=-100
)

# Print or inspect the decoded texts
for idx, text in enumerate(decoded_labels):
    print(f"Decoded Text {idx+1}: {text}\n")
