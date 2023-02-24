from typing import List

Vector = List[int]
inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

def num_differences(v1: Vector, v2: Vector) -> int:
	assert len(v1) == len(v2)
	return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0

def vector_sum(vectors: List[Vector]) -> Vector:
	assert vectors, 'Векторы не предоставлены!'
	num_elements = len(vectors[0])
	assert all(len(v) == num_elements for v in vectors), 'Разные размеры!'
	return [sum(vector[i] for vector in vectors)
			for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
	return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
	n = len(vectors)
	return scalar_multiply(1 / n, vector_sum(vectors))

def cluster_means(k: int, inputs: List[Vector], assignments: List[int]) -> List[Vector]:
	clusters =[[] for i in range(k)]
	for input, assignment in zip(inputs, assignments):
		clusters[assignment].append(input)
	return [vector_mean(cluster) if cluster else random.choice(inputs)
			for cluster in clusters]

import itertools
import random
import tqdm

def subtract(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i - w_i for v_i, w_i in zip(v, w)]

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

def squared_distance(v: Vector, w: Vector) -> float:
	return sum_of_squares(subtract(v, w))

class KMeans:
	def __init__(self, k: int) -> None:
		self.k = k
		self.means = None

	def classify(self, input: Vector) -> int:
		return min(range(self.k),
						  key = lambda i: squared_distance(input, self.means[i]))

	def train(self, inputs: List[Vector]) -> None:
		assignments = [random.randrange(self.k) for _ in inputs]
		with tqdm.tqdm(itertools.count()) as t:
			for _ in t:
				self.means = cluster_means(self.k, inputs, assignments)
				new_assignments = [self.classify(input) for input in inputs]
				num_changed = num_differences(assignments, new_assignments)
				if num_changed == 0:
					return
				assignments = new_assignments
				self.means = cluster_means(self.k, inputs, assignments)
				t.set_description(f'Changed: {num_changed} / {len(inputs)}')

random.seed(0)
clusterer = KMeans(k = 3)
clusterer.train(inputs)
means = sorted(clusterer.means)

assert len(means) == 3
assert squared_distance(means[0], [-44, 5]) < 1
assert squared_distance(means[1], [-16, -10]) < 1
assert squared_distance(means[2], [18, 20]) < 1

clusterer = KMeans(k = 2)
clusterer.train(inputs)
means = sorted(clusterer.means)

assert len(means) == 2
assert squared_distance(means[0], [-26, -5]) < 1
assert squared_distance(means[1], [18, 20]) < 1

from matplotlib import pyplot as plt

def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
	clusterer = KMeans(k)
	clusterer.train(inputs)
	means = clusterer.means
	assignments = [clusterer.classify(input) for input in inputs]
	return sum(squared_distance(input, means[cluster])
			   for input, cluster in zip(inputs, assignments))

ks = range(1, len(inputs) + 1)
errors = [squared_clustering_errors(inputs, k) for k in ks]
plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel('k')
plt.ylabel('Суммарная квадратическая ошибка')
plt.title('Суммарная ошибка против числа кластеров')
plt.show()

image_path = r'path' #Change to your path

import matplotlib.image as mpimg
img = mpimg.imread(image_path) / 256

top_row = img[0]
top_left_pixel = top_row[0]
red, green, blue = top_left_pixel

pixels = [pixel.tolist() for row in img for pixel in row]
clusterer = KMeans(5)
clusterer.train(pixels)

def recolor(pixel: Vector) -> Vector:
	cluster = clusterer.classify(pixel)
	return clusterer.means[cluster]

new_img = [[recolor(pixel) for pixel in row] for row in img]
plt.imshow(new_img)
plt.axis('off')
plt.show()

from typing import NamedTuple, Union

class Leaf(NamedTuple):
	value: Vector

leaf1 = Leaf([10, 20])
leaf2 = Leaf([30, -15])

class Merged(NamedTuple):
	children: tuple
	order: int

merged = Merged((leaf1, leaf2), order = 1)
Cluster = Union[Leaf, Merged]

def get_values(cluster: Cluster) -> List[Vector]:
	if isinstance(cluster, Leaf):
		return [cluster.value]
	else:
		return [value
				for child in cluster.children
				for value in get_values(child)]

assert get_values(merged) == [[10, 20], [30, -15]]

from typing import Callable
import math

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

def squared_distance(v: Vector, w: Vector) -> float:
	return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v, w))

def cluster_distance(cluster1: Cluster, cluster2: Cluster,
					 distance_agg: Callable = min) -> float:
	return distance_agg([distance(v1, v2)
						 for v1 in get_values(cluster1)
						 for v2 in get_values(cluster2)])

def get_merge_order(cluster: Cluster) -> float:
	if isinstance(cluster, Leaf):
		return float('inf')
	else:
		return cluster.order

from typing import Tuple

def get_children(cluster: Cluster):
	if isinstance(cluster, Leaf):
		raise TypeError('Лист не имеет дочерних элементов')
	else:
		return cluster.children

def bottom_up_cluster(inputs: List[Vector],
					  distance_agg: Callable = min) -> Cluster:
	clusters: List[Cluster] = [Leaf(input) for input in inputs]

	def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
		return cluster_distance(pair[0], pair[1], distance_agg)

	while len(clusters) > 1:
		c1, c2 = min(((cluster1, cluster2)
					  for i, cluster1 in enumerate(clusters)
					  for cluster2 in clusters[:i]),
					  key = pair_distance)
		clusters = [c for c in clusters if c != c1 and c != c2]
		merged_cluster = Merged((c1, c2), order = len(clusters))
		clusters.append(merged_cluster)

	return clusters[0]

base_cluster = bottom_up_cluster(inputs)

def generate_clusters(base_cluster: Cluster,
					  num_clusters: int) -> List[Cluster]:
	clusters = [base_cluster]
	while len(clusters) < num_clusters:
		next_cluster = min(clusters, key = get_merge_order)
		clusters = [c for c in clusters if c != next_cluster]
		clusters.extend(get_children(next_cluster))
	return clusters

three_clusters = [get_values(cluster)
				  for cluster in generate_clusters(base_cluster, 3)]

for i, cluster, marker, color in zip([1, 2, 3], three_clusters,
							 ['D', 'o', '*'], ['r', 'g', 'b']):
	xs, ys = zip(*cluster)
	plt.scatter(xs, ys, color = color, marker = marker)
	x, y = vector_mean(cluster)
	plt.plot(x, y, marker = '$' + str(i) + '$', color = 'black')

plt.title('Места проживания - 3 кластера снизу вверх, min')
plt.xlabel('Кварталы к востоку от центра города')
plt.ylabel('Кварталы к северу от центра города')
plt.show()
