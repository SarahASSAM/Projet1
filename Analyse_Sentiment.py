import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Chemin du fichier JSONL
file_path = "reviews.jsonl"

# Lire les 200 premières lignes
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i >= 200:  # Limiter à 200 avis
            break
        data.append(json.loads(line.strip()))  # Charger chaque ligne comme un objet JSON

# Extraire les notes, les textes et les titres
ratings = [review.get("rating") for review in data]
texts = [review.get("text") for review in data]
titles = [review.get("title", "") for review in data]  # "" si le titre est manquant

# Combiner les titres et les textes
combined_texts = [f"{title}. {text}" if title else text for title, text in zip(titles, texts)]

# Convertir en DataFrame
reviews_df = pd.DataFrame({"rating": ratings, "text": texts, "title": titles, "combined_text": combined_texts})

# Charger le tokenizer et le modèle BERT précédent
tokenizer_bert = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_bert = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Charger le tokenizer et le nouveau modèle
tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Fonction d'analyse des sentiments
def analyze_sentiments(tokenizer, model, texts, batch_size=16):
    # Tokenisation
    tokenized_data = tokenizer(
        texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    
    # Diviser en lots
    dataloader = DataLoader(
        dataset=list(zip(input_ids, attention_mask)),
        batch_size=batch_size,
        shuffle=False
    )
    
    predictions = []
    model.eval()  # Mode évaluation
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_masks = batch
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            predictions.extend(predicted_classes.tolist())
    
    return predictions

# Analyse avec le modèle BERT (notes de 1 à 5) sur combined_text
predicted_ratings_bert = analyze_sentiments(tokenizer_bert, model_bert, reviews_df["combined_text"].tolist(), batch_size=16)

# Analyse avec le modèle RoBERTa (convertir les classes en notes 1 à 5) sur combined_text
predicted_ratings_roberta_raw = analyze_sentiments(tokenizer_roberta, model_roberta, reviews_df["combined_text"].tolist(), batch_size=16)

# Convertir les classes de RoBERTa (0: négatif, 1: neutre, 2: positif) en notes
predicted_ratings_roberta = [1 if cls == 0 else 3 if cls == 1 else 5 for cls in predicted_ratings_roberta_raw]

# Ajouter les prédictions au DataFrame
reviews_df["predicted_rating_bert"] = predicted_ratings_bert
reviews_df["predicted_rating_roberta"] = predicted_ratings_roberta

# Évaluation des performances
correlation_bert, _ = pearsonr(reviews_df["rating"], reviews_df["predicted_rating_bert"])
correlation_roberta, _ = pearsonr(reviews_df["rating"], reviews_df["predicted_rating_roberta"])

# Afficher les corrélations
print(f"Corrélation pour le modèle BERT (avec titres) : {correlation_bert}")
print(f"Corrélation pour le modèle RoBERTa (avec titres) : {correlation_roberta}")



