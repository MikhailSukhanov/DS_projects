from typing import List, Dict, Callable
from collections import Counter
import math
import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
	return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
	return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ''):
	histogram = make_histogram(points, bucket_size)
	plt.bar(list(histogram.keys()), list(histogram.values()), width = bucket_size)
	plt.title(title)
	plt.show()

import random

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1,
					   tolerance: float = 0.00001) -> float:
	if mu != 0 or sigma != 1:
		return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance)
	low_z = -10
	hi_z = 10
	while hi_z - low_z > tolerance:
		mid_z = (low_z + hi_z) / 2
		mid_p = normal_cdf(mid_z)
		if mid_p < p:
			low_z = mid_z
		else:
			hi_z = mid_z
	return mid_z

random.seed(0)

uniform = [200 * random.random() - 100 for _ in range(10000)]
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, 'Равномерная гистограмма')
plot_histogram(normal, 10, 'Гистограмма нормального распределения')

def random_normal():
	return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker = '.', color = 'black', label = 'ys1')
plt.scatter(xs, ys2, marker = '.', color = 'gray', label = 'ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc = 9)
plt.title('Совсем разные совместные распределения')
plt.show()

def de_mean(xs: List[float]) -> List[float]:
	x_bar = mean(xs)
	return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
	assert len(xs) >= 2, 'Дисперсия требует наличия не менее двух элементов'
	n = len(xs)
	deviations = de_mean(xs)
	sum_of_squares = sum([d ** 2 for d in deviations])
	return sum_of_squares / (n - 1)

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))

def covariance(xs: List[float], ys: List[float]) -> float:
	assert len(xs) == len(ys), 'xs и ys должны иметь одинаковое число элементов'
	mean_xs = mean(xs)
	mean_ys = mean(ys)
	cov = sum([(x_i - mean_xs) * (y_i - mean_ys)
			  for x_i, y_i in zip(xs, ys)]) / (len(xs) - 1)
	return cov

def correlation(xs: List[float], ys: List[float]) -> float:
	stdev_x = standard_deviation(xs)
	stdev_y = standard_deviation(ys)
	if stdev_x > 0 and stdev_y > 0:
		return covariance(xs, ys) / stdev_x / stdev_y
	else:
		return 0

Vector = List[int]
Matrix = List[List[float]]

def make_matrix(num_rows: int, num_cols: int,
				entry_fn: Callable[[int, int], float]) -> Matrix:
	return [[entry_fn(i, j)
			 for j in range(num_cols)]
			 for i in range(num_rows)]

def correlation_matrix(data: Matrix) -> Matrix:
	def correlation_ij(i: int, j: int) -> float:
		return correlation(data[i], data[j])
	return make_matrix(len(data), len(data), correlation_ij)

def random_row() -> List[float]:
	row = [0.0, 0, 0, 0]
	row[0] = random_normal()
	row[1] = -5 * row[0] + random_normal()
	row[2] = row[0] + row[1] + 5 * random_normal()
	row[3] = 6 if row[2] > -2 else 0
	return row

num_points = 100
corr_rows = [random_row() for _ in range(num_points)]
corr_data = [list(col) for col in zip(*corr_rows)]
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
	for j in range(num_vectors):
		if i != j:
			ax[i][j].scatter(corr_data[j], corr_data[i])
		else:
			ax[i][j].annotate('Серия' + str(i), (0.5, 0.5),
							  xycoords = 'axes fraction',
							  ha = 'center', va = 'center')
		if i < num_vectors - 1:
			ax[i][j].xaxis.set_visible(False)
		if j > 0: ax[i][j].yaxis.set_visible(False)

ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

import datetime

stock_price = {'closing_price': 102.06,
               'date': datetime.date(2014, 8, 29),
               'symbol': 'AAPL'}

from collections import namedtuple

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple

class StockPrice1(NamedTuple):
	symbol: str
	date: datetime.date
	closing_price: float

	def is_high_tech(self) -> bool:
		return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price1 = StockPrice1('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price1.symbol == 'MSFT'
assert price1.closing_price == 106.03
assert price1.is_high_tech()

from dataclasses import dataclass

@dataclass
class StockPrice2:
	symbol: str
	date: datetime.date
	closing_price: float

	def is_high_tech(self) -> bool:
		return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

price2.closing_price /= 2
assert price2.closing_price == 53.015

from dateutil.parser import parse

def parse_row(row: List[str]) -> StockPrice:
	symbol, date, closing_price = row
	return StockPrice(symbol = symbol,
					  date = parse(date).date(),
					  closing_price = float(closing_price))

stock = parse_row(['MSFT', '2018-12-14', '106.03'])

assert stock.symbol == 'MSFT'
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

from typing import Optional
import re

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
	symbol, date_, closing_price_ = row
	if not re.match(r'^[A-Z]+$', symbol):
		return None
	try:
		date = parse(date_).date()
	except ValueError:
		return None
	try:
		closing_price = float(closing_price_)
	except ValueError:
		return None
	return StockPrice(symbol, date, closing_price)

assert try_parse_row(['MSFT0', '2018-12-14', '106.03']) is None
assert try_parse_row(['MSFT', '2018-12--14', '106.03']) is None
assert try_parse_row(['MSFT', '2018-12-14', 'x']) is None
assert try_parse_row(['MSFT', '2018-12-14', '106.03']) == stock

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

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v, w))

a_to_b = distance([63, 150], [67, 160])
a_to_c = distance([63, 150], [70, 171])
b_to_c = distance([67, 160], [70, 171])
print(a_to_b, a_to_c, b_to_c)

a_to_b = distance([160, 150], [170.2, 160])
a_to_c = distance([160, 150], [177.8, 171])
b_to_c = distance([170.2, 160], [177.8, 171])
print(a_to_b, a_to_c, b_to_c)

from typing import Tuple

def scalar_multiply(c: float, v: Vector) -> Vector:
	return [c * v_i for v_i in v]

def vector_sum(vectors: List[Vector]) -> Vector:
	assert vectors, 'Векторы не предоставлены!'
	num_elements = len(vectors[0])
	assert all(len(v) == num_elements for v in vectors), 'Разные размеры!'
	return [sum(vector[i] for vector in vectors)
			for i in range(num_elements)]

def vector_mean(vectors: List[Vector]) -> Vector:
	n = len(vectors)
	return scalar_multiply(1 / n, vector_sum(vectors))

def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)

def de_mean(xs: List[float]) -> List[float]:
	x_bar = mean(xs)
	return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
	assert len(xs) >= 2, 'Дисперсия требует наличия не менее двух элементов'
	n = len(xs)
	deviations = de_mean(xs)
	sum_of_squares = sum([d ** 2 for d in deviations])
	return sum_of_squares / (n - 1)

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))

def magnitude(v: Vector) -> float:
	return math.sqrt(sum_of_squares(v))

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
	dim = len(data[0])
	means = vector_mean(data)
	stdevs = [standard_deviation([vector[i] for vector in data])
			  for i in range(dim)]
	return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

def rescale(data: List[Vector]) -> List[Vector]:
	dim = len(data[0])
	means, stdevs = scale(data)
	rescaled = [v[:] for v in data]
	for v in rescaled:
		for i in range(dim):
			if stdevs[i] > 0:
				v[i] = (v[i] - means[i]) / stdevs[i]
	return rescaled

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

import tqdm

for i in tqdm.tqdm(range(100)):
	_ = [random.random() for _ in range(100000)]

def primes_up_to(n: int) -> List[int]:
	primes = [2]
	with tqdm.trange(3, n) as t:
		for i in t:
			i_is_prime = not any(i % p == 0 for p in primes)
			if i_is_prime:
				primes.append(i)
			t.set_description(f'{len(primes)} простых')
	return primes

my_primes = primes_up_to(100)

def de_mean(data: List[Vector]) -> List[Vector]:
	mean = vector_mean(data)
	return [subtract(vector, mean) for vector in data]

def direction(w: Vector) -> Vector:
	mag = magnitude(w)
	return [w_i / mag for w_i in w]

def directional_varience(data: List[Vector], w: Vector) -> float:
	w_dir = direction(w)
	return sum(dot(v, w_dir) ** 2 for v in data)

def directional_varience_gradient(data: List[Vector], w: Vector) -> Vector:
	w_dir = direction(w)
	return [sum(2 * dot(v, w_dir) * v[i] for v in data)
			for i in range(len(w))]

def add(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i + w_i for v_i, w_i in zip(v, w)]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
	assert len(v) == len(gradient)
	step = scalar_multiply(step_size, gradient)
	return add(v, step)

def first_principal_component(data: List[Vector],
							  n: int = 100,
							  step_size: float = 0.1) -> Vector:
	guess = [1.0 for _ in data[0]]
	with tqdm.trange(n) as t:
		for _ in t:
			dv = directional_varience(data, guess)
			gradient = directional_varience_gradient(data, guess)
			guess = gradient_step(guess, gradient, step_size)
			t.set_description(f'dv: {dv:.3f}')
	return direction(guess)

def project(v: Vector, w: Vector) -> Vector:
	projection_length = dot(v, w)
	return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
	return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
	return [remove_projection_from_vector(v, w) for v in data]

def pca(data: List[Vector], num_components: int) -> List[Vector]:
	components: List[Vector] = []
	for _ in range(num_components):
		component = first_principal_component(data)
		components.append(component)
		data = remove_projection(data, component)
	return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
	return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
	return [transform_vector(v, components) for v in data]
