from sklearn.metrics import silhouette_score

def evaluate_model(features, labels):
    score = silhouette_score(features, labels)
    print("Silhouette Score:", score)
