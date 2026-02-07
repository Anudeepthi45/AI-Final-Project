import pandas as pd
from data.netflix_data_loader import load_data
from preprocessing.text_preprocessing import clean_text
from model.clustering_model import build_model
from evaluation.metrics import evaluate_model

# Load dataset
df = load_data()

# Handle missing values
for col in ['description', 'listed_in', 'cast', 'director']:
    df[col] = df[col].fillna("")

# Combine text features
df['combined_features'] = (
    df['description'] + " " +
    df['listed_in'] + " " +
    df['cast'] + " " +
    df['director']
)

# Clean text
df['cleaned_text'] = df['combined_features'].apply(clean_text)

# Build clustering model
features, clusters = build_model(df['cleaned_text'])

# Assign clusters
df['cluster'] = clusters

# Evaluate
evaluate_model(features, clusters)

# Save result
df[['title', 'type', 'listed_in', 'cluster']].to_csv(
    "result/clustering_output.csv",
    index=False
)
