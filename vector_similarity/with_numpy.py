import numpy as np

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
    )

def get_most_similar_vector(search, items):
    search_vector = np.array(items[search])
    scores = {}

    for key, vector in items.items():
        if key == search:
            continue
        scores[key] = cosine_similarity(search_vector, vector)

    return max(scores, key=scores.get)

texts = {
    "apple": [1,0,0],
    "banana": [0.9,0.1,0],
    "car": [0,0,1]
}

print(get_most_similar_vector("apple", texts))