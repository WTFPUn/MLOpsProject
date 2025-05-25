from sklearn.metrics import calinski_harabasz_score

def calculate_vector_spread(vectors, labels):
    return calinski_harabasz_score(vectors, labels)
