import pickle
from nltk.translate.bleu_score import corpus_bleu
from predict import generate_caption
from tensorflow.keras.models import load_model

model = load_model("../models/caption_model.keras")
tokenizer = pickle.load(open("../models/tokenizer.pkl", "rb"))
features = pickle.load(open("../features/features.pkl", "rb"))
mapping = pickle.load(open("../features/mapping.pkl", "rb"))

actual, predicted = [], []

for key in list(mapping.keys())[:500]:
    captions = mapping[key]
    photo = features[key]

    y_pred = generate_caption(photo).split()

    refs = []
    for cap in captions:
        cap = cap.replace("startseq", "").replace("endseq", "").strip()
        refs.append(cap.split())

    actual.append(refs)
    predicted.append(y_pred)

print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))