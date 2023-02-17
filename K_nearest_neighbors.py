from typing import List, TypeVar, Tuple
from collections import Counter
import math

def raw_majority_vote(labels: List[str]) -> str:
	votes = Counter(labels)
	winner, _ = votes.most_common(1)[0]
	return winner

assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

def majority_vote(labels: List[str]) -> str:
	vote_counts = Counter(labels)
	winner, winner_count = vote_counts.most_common(1)[0]
	num_winners = len([count
					  for count in vote_counts.values()
					  if count == winner_count])
	if num_winners == 1:
		return winner
	else:
		return majority_vote(labels[:-1])

assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

from typing import NamedTuple

Vector = List[int]

def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

def subtract(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i - w_i for v_i, w_i in zip(v, w)]

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

def squared_distance(v: Vector, w: Vector) -> float:
	return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v, w))

class LabeledPoint(NamedTuple):
	point: Vector
	label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint],
				 new_point: Vector) -> str:
	by_distance = sorted(labeled_points,
						 key = lambda lp: distance(lp.point, new_point))
	k_nearest_labels = [lp.label for lp in by_distance[:k]]
	return majority_vote(k_nearest_labels)

import requests

data = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open('iris.data', 'w') as f:
	f.write(data.text)

from typing import Dict
import csv
from collections import defaultdict

def parse_iris_row(row: List[str]) -> LabeledPoint:
	measurements = [float(value) for value in row[:-1]]
	label = row[-1].split('-')[-1]
	return LabeledPoint(measurements, label)

with open('iris.data') as f:
	reader = csv.reader(f)
	iris_data = [parse_iris_row(row) for row in reader if len(row) > 0]

points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
	points_by_species[iris.label].append(iris.point)

from matplotlib import pyplot as plt

metrics = ['длн. чашелистика', 'шир. чашелистика', 'длн. лепестка', 'шир. лепестка']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x']

fig, ax = plt.subplots(2, 3)

for row in range(2):
	for col in range(3):
		i, j = pairs[3 * row + col]
		ax[row][col].set_title(f'{metrics[i]} против {metrics[j]}', fontsize = 8)
		ax[row][col].set_xticks([])
		ax[row][col].set_yticks([])
		for mark, (species, points) in zip(marks, points_by_species.items()):
			xs = [point[i] for point in points]
			ys = [point[j] for point in points]
			ax[row][col].scatter(xs, ys, marker = mark, label = species)

ax[-1][-1].legend(loc = 'lower right', prop = {'size': 6})
plt.show()

import random

X = TypeVar('X')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
	data = data[:]
	random.shuffle(data)
	cut = int(len(data) * prob)
	return data[:cut], data[cut:]

random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.7)

assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
	predicted = knn_classify(5, iris_train, iris.point)
	actual = iris.label
	if predicted == actual:
		num_correct += 1
	confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)

def random_point(dim: int) -> Vector:
	return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int) -> List[float]:
	return [distance(random_point(dim), random_point(dim))
			for _ in range(num_pairs)]

import tqdm

dimensions = range(1, 101)
avg_distances = []
min_distances = []

random.seed(0)

for dim in tqdm.tqdm(dimensions, desc = 'Проклятие размерности'):
	distances = random_distances(dim, 10000)
	avg_distances.append(mean(distances))
	min_distances.append(min(distances))

min_avg_ratio = [min_dist / avg_dist
				 for min_dist, avg_dist in zip(min_distances, avg_distances)]
