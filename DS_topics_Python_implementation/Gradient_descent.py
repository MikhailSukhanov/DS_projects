from typing import List, Callable

Vector = List[int]

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
	return (f(x + h) - f(x)) / h

def square(x: float) -> float:
	return x * x

def derivative(x: float) -> float:
	return 2 * x

xs = range(-10, 10)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h = 0.001) for x in xs]

import matplotlib.pyplot as plt

plt.title('Фактические производные и их оценки')
plt.plot(xs, actuals, 'rx', label = 'Actual')
plt.plot(xs, estimates, 'b+', label = 'Estimate')
plt.legend(loc = 9)
plt.show()

def partial_difference_quotient(f: Callable[[Vector], float],
								v: Vector,
								i: int,
								h: float) -> float:
	w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
	return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
					  v: Vector,
					  h: float = 0.0001):
	return [partial_difference_quotient(f, v, i, h)
			for i in range(len(v))]

import math, random

def squared_distance(v: Vector, w: Vector) -> float:
	return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v, w))

def add(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i - w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c: float, v: Vector) -> Vector:
	return [c * v_i for v_i in v]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
	assert len(v) == len(gradient)
	step = scalar_multiply(step_size, gradient)
	return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
	return [2 * v_i for v_i in v]

v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
	grad = sum_of_squares_gradient(v)
	v = gradient_step(v, grad, -0.01)
	print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
	slope, intercept = theta
	predicted = slope * x + intercept
	error = (predicted - y)
	squared_error = error ** 2
	grad = [2 * error * x, 2 * error]
	return grad

def vector_sum(vectors: List[Vector]) -> Vector:
	assert vectors, 'Векторы не предоставлены!'
	num_elements = len(vectors[0])
	assert all(len(v) == num_elements for v in vectors), 'Разные размеры!'
	return [sum(vector[i] for vector in vectors)
			for i in range(num_elements)]

def vector_mean(vectors: List[Vector]) -> Vector:
	n = len(vectors)
	return scalar_multiply(1 / n, vector_sum(vectors))

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001

for epoch in range(5000):
	grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
	theta = gradient_step(theta, grad, -learning_rate)

slope, intercept = theta
assert 19.9 < slope < 20.1, 'Наклон должен быть равным примерно 20'
assert 4.9 < intercept < 5.1, 'Пересечение должно быть равным примерно 5'

from typing import TypeVar, Iterator

T = TypeVar('T')

def minibatches(dataset: List[T],
				batch_size: int,
				shuffle: bool = True) -> Iterator[List[T]]:
	batch_starts = [start for start in range(0, len(dataset), batch_size)]
	if shuffle:
		random.shuffle(batch_starts)
	for start in batch_starts:
		end = start + batch_size
		yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
	for batch in minibatches(inputs, batch_size = 20):
		grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
		theta = gradient_step(theta, grad, -learning_rate)

slope, intercept = theta
assert 19.9 < slope < 20.1, 'Наклон должен быть равным примерно 20'
assert 4.9 < intercept < 5.1, 'Пересечение должно быть равным примерно 5'

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
	for x, y in inputs:
		grad = linear_gradient(x, y, theta)
		theta = gradient_step(theta, grad, -learning_rate)

slope, intercept = theta
assert 19.9 < slope < 20.1, 'Наклон должен быть равным примерно 20'
assert 4.9 < intercept < 5.1, 'Пересечение должно быть равным примерно 5'
