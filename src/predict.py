import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load model and required files
# ==============================
model = load_model("../models/caption_model.keras")
tokenizer = pickle.load(open("../models/tokenizer.pkl", "rb"))
max_length = pickle.load(open("../models/max_length.pkl", "rb"))
features = pickle.load(open("../features/features.pkl", "rb"))

index_word = tokenizer.index_word
word_index = tokenizer.word_index


# ==============================
# Beam Search with Improvements
# ==============================
def generate_caption(photo, beam_width=3, temperature=0.7):
    start_token = word_index.get('startseq')
    end_token = word_index.get('endseq')

    sequences = [[[start_token], 0.0]]  # [sequence, score]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)

            preds = model.predict([photo, padded], verbose=0)[0]

            # ---- Temperature sampling (adds variety) ----
            preds = np.log(preds + 1e-10) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))

            # Top candidate words
            top_ids = np.argsort(preds)[-beam_width:]

            for idx in top_ids:
                prob = preds[idx]

                # ---- Repetition penalty ----
                repeat_penalty = 1.2 if idx in seq else 1.0

                candidate_seq = seq + [idx]
                candidate_score = score - np.log(prob + 1e-10) * repeat_penalty

                all_candidates.append([candidate_seq, candidate_score])

        # Keep best sequences
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]

    # Convert tokens → words
    words = []
    for idx in best_seq:
        word = index_word.get(idx)
        if not word:
            continue
        if word == 'endseq':
            break
        if word != 'startseq':
            words.append(word)

    return " ".join(words)


# ==============================
# Caption Cleaning & Filtering
# ==============================
def clean_caption(text):
    if not text:
        return ""

    words = text.split()

    # Remove consecutive repeated words
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)

    text = " ".join(cleaned)

    # Replace common wrong phrases
    replacements = {
        "man in red shirt": "person",
        "two men": "two people",
        "a man": "a person",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove overused bad phrases
    bad_phrases = [
        "jumping off of ramp",
        "jumping off rock into the water",
        "jumping into the water",
    ]

    for p in bad_phrases:
        text = text.replace(p, "")

    # Clean extra spaces
    text = " ".join(text.split())

    # Capitalize first letter
    if len(text) > 0:
        text = text[0].upper() + text[1:]

    return text.strip()


# ==============================
# Test with one image
# ==============================
key = list(features.keys())[0]
photo = features[key]

caption = generate_caption(photo)
caption = clean_caption(caption)

print("Generated Caption:", caption)