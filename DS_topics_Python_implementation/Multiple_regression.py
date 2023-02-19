from typing import List, Tuple

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]

Vector = List[int]

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

def predict(x: Vector, beta: Vector) -> float:
	return dot(x, beta)

def error(x: Vector, y: float, beta: Vector) -> float:
	return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
	return error(x, y, beta) ** 2

x = [1, 2, 3]
y = 30
beta = [4, 4, 4]

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
	err = error(x, y, beta)
	return [2 * err * x_i for x_i in x]

assert sqerror_gradient(x, y, beta) == [-12, -24, -36]

import random
import tqdm

def add(v: Vector, w: Vector) -> Vector:
	assert len(v) == len(w), 'Векторы должны иметь одинаковую длину'
	return [v_i + w_i for v_i, w_i in zip(v, w)]

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

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
	assert len(v) == len(gradient)
	step = scalar_multiply(step_size, gradient)
	return add(v, step)

def least_squares_fit(xs: List[Vector],
					  ys: List[float],
					  learning_rate: float = 0.001,
					  num_steps: int = 1000,
					  batch_size: int = 1) -> Vector:
	guess = [random.random() for _ in xs[0]]
	for _ in tqdm.trange(num_steps, desc = 'least sqares fit'):
		for start in range(0, len(xs), batch_size):
			batch_xs = xs[start:start + batch_size]
			batch_ys = ys[start:start + batch_size]
			gradient = vector_mean([sqerror_gradient(x, y, guess)
								   for x, y in zip(batch_xs, batch_ys)])
			guess = gradient_step(guess, gradient, -learning_rate)
	return guess

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,
			   13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,
			   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,
			   7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			   6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,
			   4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,
			   2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
			   1,1]

daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,
				 54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,
				 32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,
				 26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,
				 25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,
				 29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,
				 37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,
				 26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,
				 34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,
				 32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,
				 20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,
				 33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,
				 19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,
				 37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,
				 22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,
				 18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,
				 30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,
				 24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,
				 8.38,27.81,32.35,23.84]

outlier = num_friends.index(100)
daily_minutes_good = [x for i, x in enumerate(daily_minutes) if i != outlier]

random.seed(0)

learning_rate = 0.001
beta = least_squares_fit(inputs, daily_minutes_good,
						 learning_rate, 5000, 25)

assert 30.5 < beta[0] < 30.7
assert 0.96 < beta[1] < 1.0
assert -1.89 < beta[2] < -1.85
assert 0.91 < beta[3] < 0.94

def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)

def de_mean(xs: List[float]) -> List[float]:
	x_bar = mean(xs)
	return [x - x_bar for x in xs]

def total_sum_of_squares(y: Vector) -> float:
	return sum(v ** 2 for v in de_mean(y))

def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
	sum_of_squared_errors = sum(error(x, y, beta) ** 2
								for x, y in zip(xs, ys))
	return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)

assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

from typing import TypeVar, Callable
import math

X = TypeVar('X')
Stat = TypeVar('Stat')

def bootstrap_sample(data: List[X]) -> List[X]:
	return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
						stats_fn: Callable[[List[X]], Stat],
						num_samples: int) -> List[Stat]:
	return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

close_to_100 = [99.5 + random.random() for _ in range(101)]
far_from_100 = ([99.5 + random.random()] +
				[random.random() for _ in range(50)] +
				[200 + random.random() for _ in range(50)])

def _median_odd(xs: List[float]) -> float:
	return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
	sorted_xs = sorted(xs)
	hi_midpoint = len(xs) // 2
	return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
	return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

def variance(xs: List[float]) -> float:
	assert len(xs) >= 2, 'Дисперсия требует наличия не менее двух элементов'
	n = len(xs)
	deviations = de_mean(xs)
	sum_of_squares = sum([d ** 2 for d in deviations])
	return sum_of_squares / (n - 1)

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_far = bootstrap_statistic(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90

import datetime

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
	x_sample = [x for x, _ in pairs]
	y_sample = [y for _, y in pairs]
	beta = least_squares_fit(x_sample, y_sample, learning_rate, 1000, 25)
	print('Бутстраповская выборка', beta)
	return beta

bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
									  estimate_sample_beta, 100)
bootstrap_standard_errors = [
	standard_deviation([beta[i] for beta in bootstrap_betas])
	for i in range(4)]

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
	if beta_hat_j > 0:
		return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
	else:
		return 2 * normal_cdf(beta_hat_j / sigma_hat_j)

assert p_value(30.58, 1.27) < 0.001
assert p_value(0.972, 0.103) < 0.001
assert p_value(-1.865, 0.155) < 0.001
assert p_value(-0.923, 1.249) > 0.4

def ridge_penalty(beta: Vector, alpha: float) -> float:
	return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector, y: float, beta: Vector, alpha: float) -> float:
	return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
	return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector, y: float, beta: Vector, alpha: float) -> Vector:
	return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))

def least_squares_fit_ridge(xs: List[Vector],
					  ys: List[float],
					  alpha: float,
					  learning_rate: float = 0.001,
					  num_steps: int = 1000,
					  batch_size: int = 1) -> Vector:
	guess = [random.random() for _ in xs[0]]
	for _ in tqdm.trange(num_steps, desc = 'least sqares fit'):
		for start in range(0, len(xs), batch_size):
			batch_xs = xs[start:start + batch_size]
			batch_ys = ys[start:start + batch_size]
			gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
								   for x, y in zip(batch_xs, batch_ys)])
			guess = gradient_step(guess, gradient, -learning_rate)
	return guess

beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0,
								 learning_rate, 5000, 25)

assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69

beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1,
								   learning_rate, 5000, 25)

assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1,
								 learning_rate, 5000, 25)

assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good, 10,
								learning_rate, 5000, 25)

assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6
