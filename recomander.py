import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

# Charger les données
import jsonlines
reviews_file = "reviews.jsonl"
reviews = []
with jsonlines.open(reviews_file) as reader:
    for obj in reader:
        reviews.append(obj)

documents = [review['text'] for review in reviews]

# Générer des embeddings avec un modèle avancé
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(documents)

# Clustering avec HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10, metric='euclidean')
hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings)

# Création d'un DataFrame pour analyser les résultats
clustered_documents = pd.DataFrame({
    'document': documents,
    'cluster': hdbscan_labels
})

# Affichage des résultats par cluster
print("Clusters générés avec HDBSCAN :")
for cluster_id in sorted(clustered_documents['cluster'].unique()):
    cluster_docs = clustered_documents[clustered_documents['cluster'] == cluster_id]
    print(f"Cluster {cluster_id} : {len(cluster_docs)} documents")
    print(cluster_docs['document'].head(3).tolist())  # Affichez les 3 premiers documents par cluster
    print()

# Visualisation avec PCA
pca_embeddings = PCA(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(10, 7))
unique_clusters = set(hdbscan_labels)
for cluster in unique_clusters:
    color = 'black' if cluster == -1 else None
    label = 'Bruit' if cluster == -1 else f'Cluster {cluster}'
    cluster_points = pca_embeddings[hdbscan_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label, alpha=0.6)

plt.title("Clustering avec HDBSCAN (PCA)")
plt.xlabel("Composante PCA 1")
plt.ylabel("Composante PCA 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Analyse des mots fréquents par cluster
def get_top_words_per_cluster(clustered_docs, cluster_labels, n=10):
    top_words = {}
    for cluster in set(cluster_labels):
        if cluster == -1:  # Ignorer le bruit
            continue
        cluster_texts = clustered_docs[clustered_docs['cluster'] == cluster]['document']
        words = " ".join(cluster_texts).split()
        word_counts = Counter(words)
        top_words[cluster] = word_counts.most_common(n)
    return top_words

top_words = get_top_words_per_cluster(clustered_documents, hdbscan_labels)

print("Mots fréquents par cluster :")
for cluster, words in top_words.items():
    print(f"Cluster {cluster} :")
    for word, count in words:
        print(f"  {word} : {count}")
    print()
