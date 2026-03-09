import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

# Load captions file
def load_captions(filepath):
    with open(filepath, "r") as file:
        captions = file.read()

    mapping = {}
    lines = captions.split("\n")[1:]  # skip header

    for line in lines:
        if len(line.strip()) == 0:
            continue

        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        image_name, caption = parts
        image_id = image_name.split(".")[0]

        if image_id not in mapping:
            mapping[image_id] = []

        mapping[image_id].append(caption)

    return mapping


# Clean captions
def clean_captions(mapping):
    for key in mapping:
        cleaned = []
        for caption in mapping[key]:
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            words = caption.split()
            words = [word for word in words if len(word) > 1 and word.isalpha()]
            caption = "startseq " + " ".join(words) + " endseq"
            cleaned.append(caption)

        mapping[key] = cleaned

    return mapping


# Tokenizer
def create_tokenizer(mapping):
    all_captions = []
    for key in mapping:
        all_captions.extend(mapping[key])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)

    return tokenizer, vocab_size, max_length