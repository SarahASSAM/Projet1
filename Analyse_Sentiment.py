import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ///// Charger les données
# Charger et préparer les données
data = pd.read_json("reviews.jsonl", lines=True)
data['combined_text'] = data['title'] + " " + data['text']
reviews = data[['rating', 'combined_text']].dropna()

# Limiter les données pour les tests
reviews = reviews.head(200)

# ///// Charger le modèle et tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Détecter le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ///// Prétraitement et analyse des sentiments par lots
# Créer une classe Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Créer le DataLoader
batch_size = 16
dataset = ReviewDataset(reviews['combined_text'].tolist())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Analyser les sentiments par lots
all_predictions = []
for batch in dataloader:
    tokens = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=-1) + 1
        all_predictions.extend(predictions)

# Ajouter les prédictions aux données
reviews['predicted_rating'] = all_predictions

# ///// Évaluation des performances
# Calculer la corrélation de Pearson
actual_ratings = reviews['rating']
predicted_ratings = reviews['predicted_rating']
correlation, _ = pearsonr(actual_ratings, predicted_ratings)
print(f"Corrélation de Pearson : {correlation}")

# Afficher les premières lignes avec les prédictions
print(reviews[['rating', 'predicted_rating', 'combined_text']].head(10))

# ///// Visualisation des résultats
# Matrice de corrélation
plt.figure(figsize=(10, 8))
correlation_matrix = reviews[['rating', 'predicted_rating']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt='.2f')
plt.title("Corrélation entre les Notes Réelles et Prédites", fontsize=16)
plt.xlabel("Variables", fontsize=14)
plt.ylabel("Variables", fontsize=14)
plt.tight_layout()
plt.show()

# Matrice de confusion
confusion_matrix = pd.crosstab(reviews['rating'], reviews['predicted_rating'], rownames=['Vraies Notes'], colnames=['Notes Prédites'], normalize=False)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title("Matrice de Confusion : Vraies Notes vs Notes Prédites", fontsize=16)
plt.xlabel("Notes Prédites", fontsize=14)
plt.ylabel("Vraies Notes", fontsize=14)
plt.tight_layout()
plt.show()

# Graphique des fréquences
plt.figure(figsize=(12, 8))
bar_width = 0.4
ratings_counts = reviews['rating'].value_counts().sort_index()
predicted_counts = reviews['predicted_rating'].value_counts().sort_index()
x_real = [x - bar_width / 2 for x in ratings_counts.index]
x_predicted = [x + bar_width / 2 for x in predicted_counts.index]

plt.bar(x_real, ratings_counts, width=bar_width, label='Vraies Notes', alpha=0.8, color='teal')
plt.bar(x_predicted, predicted_counts, width=bar_width, label='Notes Prédites', alpha=0.8, color='salmon')
plt.xlabel('Notes (1 à 5)', fontsize=14)
plt.ylabel('Fréquence', fontsize=14)
plt.title('Comparaison des Fréquences : Vraies Notes vs Prédites', fontsize=16)
plt.xticks([1, 2, 3, 4, 5], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
