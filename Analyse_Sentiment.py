
import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# # Chemin du fichier JSONL
# file_path = "reviews.jsonl"

# # Lire les 200 premières lignes
# data = []
# with open(file_path, "r", encoding="utf-8") as file:
#     for i, line in enumerate(file):
#         if i >= 200:  # Limiter à 200 avis
#             break
#         data.append(json.loads(line.strip()))  # Charger chaque ligne comme un objet JSON

# # Extraire les notes et les textes
# ratings = [review.get("rating") for review in data]
# texts = [review.get("text") for review in data]

# # Convertir en DataFrame
# reviews_df = pd.DataFrame({"rating": ratings, "text": texts})

# # Charger le tokenizer et le modèle
# tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# # Tokenisation et DataLoader
# batch_size = 16  # Ajustez selon vos ressources
# tokenized_data = tokenizer(
#     texts,
#     max_length=512,
#     padding="max_length",
#     truncation=True,
#     return_tensors="pt"
# )

# # Préparer les données pour le DataLoader
# input_ids = tokenized_data["input_ids"]
# attention_masks = tokenized_data["attention_mask"]
# labels = torch.tensor(ratings)

# # Créer le DataLoader
# dataloader = DataLoader(
#     dataset=list(zip(input_ids, attention_masks, labels)),
#     batch_size=batch_size,
#     shuffle=False
# )


# # Analyse des sentiments
# predictions = []
# model.eval()  # Mode évaluation
# with torch.no_grad():  # Désactiver la rétropropagation pour économiser les ressources
#     for batch in dataloader:
#         batch_input_ids, batch_attention_masks, _ = batch
#         # Passage dans le modèle
#         outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
#         logits = outputs.logits

#         # Appliquer softmax pour obtenir les probabilités
#         probabilities = F.softmax(logits, dim=1)

#         # Trouver les classes avec les probabilités maximales
#         predicted_classes = torch.argmax(probabilities, dim=1) + 1  # Classes de 1 à 5
#         predictions.extend(predicted_classes.tolist())

# # Ajouter les prédictions au DataFrame
# reviews_df["predicted_rating"] = predictions

# # Évaluation des performances
# # Corrélation de Pearson entre les notes réelles et les prédictions
# real_ratings = reviews_df["rating"]
# predicted_ratings = reviews_df["predicted_rating"]
# correlation, _ = pearsonr(real_ratings, predicted_ratings)

# # Afficher les résultats
# print("Corrélation de Pearson entre les notes réelles et prédites :", correlation)
# print("Aperçu des résultats avec prédictions :")
# print(reviews_df.head())

# # Sauvegarder les résultats dans un fichier CSV
# reviews_df.to_csv("reviews_with_sentiments_and_correlation.csv", index=False)
# print("Les résultats ont été sauvegardés dans 'reviews_with_sentiments_and_correlation.csv'.")


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

# Analyse avec le modèle BERT (notes de 1 à 5)
predicted_ratings_bert = analyze_sentiments(tokenizer_bert, model_bert, reviews_df["text"].tolist(), batch_size=16)

# Analyse avec le modèle RoBERTa (convertir les classes en notes 1 à 5)
predicted_ratings_roberta_raw = analyze_sentiments(tokenizer_roberta, model_roberta, reviews_df["text"].tolist(), batch_size=16)

# Convertir les classes de RoBERTa (0: négatif, 1: neutre, 2: positif) en notes
predicted_ratings_roberta = [1 if cls == 0 else 3 if cls == 1 else 5 for cls in predicted_ratings_roberta_raw]

# Ajouter les prédictions au DataFrame
reviews_df["predicted_rating_bert"] = predicted_ratings_bert
reviews_df["predicted_rating_roberta"] = predicted_ratings_roberta

# Évaluation des performances
correlation_bert, _ = pearsonr(reviews_df["rating"], reviews_df["predicted_rating_bert"])
correlation_roberta, _ = pearsonr(reviews_df["rating"], reviews_df["predicted_rating_roberta"])

# Afficher les corrélations
print(f"Corrélation pour le modèle BERT : {correlation_bert}")
print(f"Corrélation pour le modèle RoBERTa : {correlation_roberta}")

# Comparaison des distributions
plt.figure(figsize=(12, 6))

# Histogramme pour les prédictions BERT
plt.hist(reviews_df["predicted_rating_bert"], bins=5, alpha=0.6, label="Modèle BERT", edgecolor="black")

# Histogramme pour les prédictions RoBERTa
plt.hist(reviews_df["predicted_rating_roberta"], bins=5, alpha=0.6, label="Modèle RoBERTa", edgecolor="black")

# Ajouter des labels et une légende
plt.title("Comparaison des prédictions entre les modèles BERT et RoBERTa")
plt.xlabel("Notes (1 à 5)")
plt.ylabel("Fréquence")
plt.legend(loc="upper left")

# Afficher le graphique
plt.show()

# Sauvegarder les résultats
reviews_df.to_csv("reviews_with_sentiments_comparison.csv", index=False)
print("Les résultats ont été sauvegardés dans 'reviews_with_sentiments_comparison.csv'.")


##########Comparer notes réelles et prédictions ########
import matplotlib.pyplot as plt

# Comparer les distributions des notes réelles et des prédictions pour le modèle BERT
plt.figure(figsize=(10, 6))

# Histogramme des notes réelles
plt.hist(reviews_df["rating"], bins=5, alpha=0.6, label="Notes réelles", edgecolor="black", color="blue")

# Histogramme des prédictions BERT
plt.hist(reviews_df["predicted_rating_bert"], bins=5, alpha=0.6, label="Prédictions BERT", edgecolor="black", color="green")

# Ajouter des labels et une légende
plt.title("Comparaison des distributions des notes réelles et prédites (BERT)")
plt.xlabel("Notes (1 à 5)")
plt.ylabel("Fréquence")
plt.legend(loc="upper left")

# Afficher le graphique
plt.show()
