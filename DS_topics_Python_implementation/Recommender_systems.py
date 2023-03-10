users_interests = [
    ['Hadoop', 'Big Data', 'HBase', 'Java', 'Spark', 'Storm', 'Cassandra'],
    ['NoSQL', 'MongoDB', 'Cassandra', 'HBase', 'Postgres'],
    ['Python', 'scikit-learn', 'scipy', 'numpy', 'statsmodels', 'pandas'],
    ['R', 'Python', 'statistics', 'regression', 'probability'],
    ['machine learning', 'regression', 'decision trees', 'libsvm'],
    ['Python', 'R', 'Java', 'C++','Haskell', 'programming languages'],
    ['statistics', 'probability', 'mathematics', 'theory'],
    ['machine learning', 'scikit-learn', 'Mahout', 'neural networks'],
    ['neural networks', 'deep learning', 'Big Data', 'artificial intelligence'],
    ['Hadoop', 'Java', 'MapReduce', 'Big Data'],
    ['statistics', 'R', 'statsmodels'],
    ['C++', 'deep learning', 'artificial intelligence', 'probability'],
    ['pandas', 'R', 'Python'],
    ['databases', 'HBase', 'Postgres', 'MySQL', 'MongoDB'],
    ['libsvm', 'regression', 'support vector machines']]

from collections import Counter

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

from typing import List, Tuple

def most_popular_new_interests(user_interests: List[str],
                               max_results: int = 5) -> List[Tuple[str, int]]:
    suggestions = [(interest, frequency)
                   for interest, frequency
                   in popular_interests.most_common()
                   if interest not in user_interests]
    return suggestions[:max_results]

unique_interests = sorted(list({interest
                                for user_interests in users_interests
                                for interest in user_interests}))

def make_user_interest_vector(user_interests: List[str]) -> List[int]:
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

user_interest_vectors = [make_user_interest_vector(user_interests)
                         for user_interests in users_interests]

import math

Vector = List[int]

def dot(v: Vector, w: Vector) -> float:
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_vectors]
                     for interest_vector_i in user_interest_vectors]

assert 0.56 < user_similarities[0][9] < 0.58
assert 0.18 < user_similarities[0][8] < 9.20

def most_similar_users_to(user_id: int) -> List[Tuple[int, float]]:
    pairs = [(other_user_id, similarity)
             for other_user_id, similarity
             in enumerate(user_similarities[user_id])
             if user_id != other_user_id and similarity > 0]
    return sorted(pairs, key = lambda pair: pair[-1], reverse = True)

print(most_similar_users_to(0))

from collections import defaultdict

def user_based_suggestions(user_id: int, include_current_interests: bool = False):
    suggestions: Dict[str, float] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key = lambda pair: pair[-1], reverse = True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(user_based_suggestions(0))

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_vectors]
                        for j, _ in enumerate(unique_interests)]
interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

def most_similar_interests_to(interest_id: int):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs, key = lambda pair: pair[-1], reverse = True)

def item_based_suggestions(user_id: int, include_current_interests: bool = False):
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key = lambda pair: pair[-1], reverse = True)
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]
