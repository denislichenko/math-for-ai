import math

def get_l2_norm(vector):
    return math.sqrt(sum(x**2 for x in vector))

def get_normalized_vector(vector):
    norm = get_l2_norm(vector)
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def dot_product(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

def cosine_similarity(vector1, vector2):
    return dot_product(
        get_normalized_vector(vector1),
        get_normalized_vector(vector2)
    )

def get_most_similar_vector(search, items):
    search_vector = items.get(search)
    similarity = {}

    for key, vector in items.items():
        if key == search:
            continue

        similarity[key] = cosine_similarity(search_vector, vector)

    max_similarity = max(similarity.values())

    return [key for key in similarity if similarity[key] == max_similarity][0]

texts = {
    "apple": [1,0,0],
    "banana": [0.9,0.1,0],
    "car": [0,0,1]
}

print(get_most_similar_vector("apple", texts))