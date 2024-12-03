
import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Chemin du fichier JSONL
file_path = "reviews.jsonl"

# Lire les 200 premières lignes
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i >= 200:  # Limiter à 200 avis
            break
        data.append(json.loads(line.strip()))  # Charger chaque ligne comme un objet JSON

# Extraire les notes et les textes
ratings = [review.get("rating") for review in data]
texts = [review.get("text") for review in data]

# Convertir en DataFrame
reviews_df = pd.DataFrame({"rating": ratings, "text": texts})

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Tokenisation et DataLoader
batch_size = 16  # Ajustez selon vos ressources
tokenized_data = tokenizer(
    texts,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# Préparer les données pour le DataLoader
input_ids = tokenized_data["input_ids"]
attention_masks = tokenized_data["attention_mask"]
labels = torch.tensor(ratings)

# Créer le DataLoader
dataloader = DataLoader(
    dataset=list(zip(input_ids, attention_masks, labels)),
    batch_size=batch_size,
    shuffle=False
)


# Analyse des sentiments
predictions = []
model.eval()  # Mode évaluation
with torch.no_grad():  # Désactiver la rétropropagation pour économiser les ressources
    for batch in dataloader:
        batch_input_ids, batch_attention_masks, _ = batch
        # Passage dans le modèle
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        logits = outputs.logits

        # Appliquer softmax pour obtenir les probabilités
        probabilities = F.softmax(logits, dim=1)

        # Trouver les classes avec les probabilités maximales
        predicted_classes = torch.argmax(probabilities, dim=1) + 1  # Classes de 1 à 5
        predictions.extend(predicted_classes.tolist())

# Ajouter les prédictions au DataFrame
reviews_df["predicted_rating"] = predictions
