from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

def build_model(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(text_data)

    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_features = svd.fit_transform(tfidf_matrix)

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)

    return reduced_features, clusters
