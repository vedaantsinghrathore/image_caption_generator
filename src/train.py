import pickle
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from preprocess import load_captions, clean_captions, create_tokenizer
from feature_extractor import extract_features
from model import define_model

# Paths
CAPTION_PATH = "../dataset/captions.txt"
IMAGE_PATH = "../dataset/Images/"

# Load and preprocess captions
mapping = load_captions(CAPTION_PATH)
mapping = clean_captions(mapping)

tokenizer, vocab_size, max_length = create_tokenizer(mapping)

# Extract image features
features = extract_features(IMAGE_PATH)

# Save features
pickle.dump(features, open("../features/features.pkl", "wb"))
pickle.dump(tokenizer, open("../models/tokenizer.pkl", "wb"))
pickle.dump(max_length, open("../models/max_length.pkl", "wb"))

# Data generator
def data_generator(mapping, features, tokenizer, max_length, vocab_size):
    while True:
        for key, captions in mapping.items():
            feature = features[key][0]

            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    yield ((feature, in_seq), out_seq)

# Define model
model = define_model(vocab_size, max_length)

steps = len(mapping)

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(mapping, features, tokenizer, max_length, vocab_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(4096,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32),
    ),
).batch(64)

# Train
model.fit(dataset, epochs=30, steps_per_epoch=steps)

# Save model
model.save("../models/caption_model.keras")
print("Model saved successfully.")