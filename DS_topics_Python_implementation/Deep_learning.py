Tensor = list

from typing import List

def shape(tensor: Tensor) -> List[int]:
	sizes: List[int] = []
	while isinstance(tensor, list):
		sizes.append(len(tensor))
		tensor = tensor[0]
	return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]

def is_1d(tensor: Tensor) -> bool:
	return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])

def tensor_sum(tensor: Tensor) -> float:
	if is_1d(tensor):
		return sum(tensor)
	else:
		return sum(tensor_sum(tensor_i) for tensor_i in tensor)

assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10

from typing import Callable

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
	if is_1d(tensor):
		return [f(x) for x in tensor]
	else:
		return [tensor_apply(f, tensor_i) for tensor_i in tensor]

assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
	return tensor_apply(lambda _: 0.0, tensor)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

def tensor_combine(f: Callable[[float, float], float],
				   t1: Tensor, t2: Tensor) -> Tensor:
	if is_1d(t1):
		return [f(x, y) for x, y in zip(t1, t2)]
	else:
		return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

import operator

assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]

from typing import Iterable, Tuple

class Layer:
	def forward(self, imput):
		raise NotImplementError

	def backward(self, gradient):
		raise NotImplementError

	def params(self) -> Iterable[Tensor]:
		return ()

	def grads(self) -> Iterable[Tensor]:
		return ()

def sigmoid(t: float) -> float:
	return 1 / (1 + math.exp(-t))

class Sigmoid(Layer):
	def forward(self, input: Tensor) -> Tensor:
		self.sigmoids = tensor_apply(sigmoid, input)
		return self.sigmoids

	def backward(self, gradient: Tensor) -> Tensor:
		return tensor_combine(lambda sig, grad: sig * (1 - grad) * grad,
							  self.sigmoids, gradient)

import random
import math

Vector = List[int]

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

def random_uniform(*dims: int) -> Tensor:
	if len(dims) == 1:
		return [random.random() for _ in range(dims[0])]
	else:
		return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
	if len(dims) == 1:
		return [mean + variance * inverse_normal_cdf(random.random())
				for _ in range(dims[0])]
	else:
		return [random_normal(*dims[1:], mean = mean, variance = variance)
				for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean = 10)) == [5, 6]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
	if init == 'normal':
		return random_normal(*dims)
	elif init == 'uniform':
		return random_uniform(*dims)
	elif init == 'xavier':
		variance = len(dims) / sum(dims)
		return random_normal(*dims, variance = variance)
	else:
		raise ValueError(f'unknown init: {init}')

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w)
	return sum(v_i * w_i for v_i, w_i in zip(v, w))

class Linear(Layer):
	def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.w = random_tensor(output_dim, input_dim, init = init)
		self.b = random_tensor(output_dim, init = init)

	def forward(self, input: Tensor) -> Tensor:
		self.input = input
		return [dot(input, self.w[o]) + self.b[o]
				for o in range(self.output_dim)]

	def backward(self, gradient: Tensor) -> Tensor:
		self.b_grad = gradient
		self.w_grad = [[self.input[i] * gradient[o] for i in range(self.input_dim)]
					   for o in range(self.output_dim)]
		return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
				for i in range(self.input_dim)]

	def params(self) -> Iterable[Tensor]:
		return [self.w, self.b]

	def grads(self) -> Iterable[Tensor]:
		return [self.w_grad, self.b_grad]

class Sequential(Layer):
	def __init__(self, layers: List[Layer]) -> None:
		self.layers = layers

	def forward(self, input):
		for layer in self.layers:
			input = layer.forward(input)
		return input

	def backward(self, gradient):
		for layer in reversed(self.layers):
			gradient = layer.backward(gradient)
		return gradient

	def params(self) -> Iterable[Tensor]:
		return (param for layer in self.layers for param in layer.params())

	def grads(self) -> Iterable[Tensor]:
		return (grad for layer in self.layers for grad in layer.grads())

xor_net = Sequential([Linear(input_dim = 2, output_dim = 2),
					  Sigmoid(),
					  Linear(input_dim = 2, output_dim = 1),
					  Sigmoid()])

class Loss:
	def loss(self, predicted: Tensor, actual: Tensor) -> float:
		raise NotImplementedError

	def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
		raise NotImplementedError

class SSE(Loss):
	def loss(self, predicted: Tensor, actual: Tensor) -> float:
		squared_errors = tensor_combine(
							lambda predicted, actual: (predicted - actual) ** 2,
							predicted, actual)
		return tensor_sum(squared_errors)

	def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
		return tensor_combine(
				lambda predicted, actual: 2 * (predicted - actual),
				predicted, actual)

class Optimizer:
	def step(self, layer: Layer) -> None:
		raise NotImplementedError

class GradientDescent(Optimizer):
	def __init__(self, learning_rate: float = 0.1) -> None:
		self.lr = learning_rate

	def step(self, layer: Layer) -> None:
		for param, grad in zip(layer.params(), layer.grads()):
			param[:] = tensor_combine(
						lambda param, grad: param - grad * self.lr,
						param, grad)

class Momentum(Optimizer):
	def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
		self.lr = learning_rate
		self.mo = momentum
		self.updates: List[Tensor] = []

	def step(self, layer: Layer) -> None:
		if not self.updates:
			self.updates = [zeros_like(grad) for grad in layer.grads()]
		for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
			update[:] = tensor_combine(lambda u, g: self.mo * u + (1 - self.mo) * g,
									   update, grad)
			param[:] = tensor_combine(lambda p, u: p - self.lr * u,
									  param, update)

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [[0], [1], [1], [0]]

random.seed()

net = Sequential([Linear(input_dim = 2, output_dim = 2),
				  Sigmoid(),
				  Linear(input_dim = 2, output_dim = 1)])

import tqdm

optimizer = GradientDescent(learning_rate = 0.1)
loss = SSE()

with tqdm.trange(1000) as t:
	for epoch in t:
		epoch_loss = 0
		for x, y in zip(xs, ys):
			predicted = net.forward(x)
			epoch_loss += loss.loss(predicted, y)
			gradient = loss.gradient(predicted, y)
			net.backward(gradient)
			optimizer.step(net)
		t.set_description(f'xor потеря {epoch_loss:.3f}')

for param in net.params():
	print(param)

def tanh(x: float) -> float:
	if x < -100: return -1
	elif x > 100: return 1
	em2x = math.exp(-2 * x)
	return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
	def forward(self, input: Tensor) -> Tensor:
		self.tanh = tensor_apply(tanh, input)
		return self.tanh

	def backward(self, gradient: Tensor) -> Tensor:
		return tensor_combine(lambda tanh, grad: (1 - tanh ** 2) * grad,
							  self.tanh, gradient)

class Relu(Layer):
	def forward(self, input: Tensor) -> Tensor:
		self.input = input
		return tensor_apply(lambda x: max(x, 0), input)

	def backward(self, gradient: Tensor) -> Tensor:
		return tensor_combine(lambda x, grad: grad if x > 0 else 0,
							  self.input, gradient)

def binary_encode(x: int) -> Vector:
	binary: List[float] = []
	for i in range(10):
		binary.append(x % 2)
		x //= 2
	return binary

def fizz_buzz_encode(x: int) -> Vector:
	if x % 15 == 0:
		return [0, 0, 0, 1]
	elif x % 5 == 0:
		return [0, 0, 1, 0]
	elif x % 3 == 0:
		return [0, 1, 0, 0]
	else:
		return [1, 0, 0, 0]

def argmax(xs: list) -> int:
	return max(range(len(xs)), key = lambda i: xs[i])

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

random.seed(0)

net = Sequential([Linear(input_dim = 10, output_dim = NUM_HIDDEN, init = 'uniform'),
				  Tanh(),
				  Linear(input_dim = NUM_HIDDEN, output_dim = 4, init = 'uniform'),
				  Sigmoid()])

def fizzbuzz_accuracy(low: int, hi: int, net: Layer) -> float:
	num_correct = 0
	for n in range(low, hi):
		x = binary_encode(n)
		predicted = argmax(net.forward(x))
		actual = argmax(fizz_buzz_encode(n))
		if predicted == actual:
			num_correct += 1
	return num_correct / (hi - low)

optimizer = Momentum(learning_rate = 0.1, momentum = 0.9)
loss = SSE()

with tqdm.trange(1000) as t:
	for epoch in t:
		epoch_loss = 0
	for x, y in zip(xs, ys):
		predicted = net.forward(x)
		epoch_loss += loss.loss(predicted, y)
		gradient = loss.gradient(predicted, y)
		net.backward(gradient)
		optimizer.step(net)
	accuracy = fizzbuzz_accuracy(101, 1024, net)
	t.set_description(f'fb потеря: {epoch_loss:.2f} точность: {accuracy:.2f}')

print('Тестовые результаты', fizzbuzz_accuracy(1, 101, net))

def softmax(tensor: Tensor) -> Tensor:
	if is_1d(tensor):
		largest = max(tensor)
		exps = [math.exp(x - largest) for x in tensor]
		sum_of_exps = sum(exps)
		return [exp_i / sum_of_exps for exp_i in exps]
	else:
		return [softmax(tensor_i) for tensor_i in tensor]

class SoftmaxCrossEntropy(Loss):
	def loss(self, predicted: Tensor, actual: Tensor) -> float:
		probabilities = softmax(predicted)
		likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
									probabilities, actual)
		return -tensor_sum(likelihoods)

	def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
		probabilities = softmax(predicted)
		return tensor_combine(lambda p, actual: p - actual,
							  probabilities, actual)

net = Sequential([Linear(input_dim = 10, output_dim = NUM_HIDDEN, init = 'uniform'),
				  Tanh(),
				  Linear(input_dim = NUM_HIDDEN, output_dim = 4, init = 'uniform')])

optimizer = Momentum(learning_rate = 0.1, momentum = 0.9)
loss = SoftmaxCrossEntropy()

with tqdm.trange(100) as t:
	for epoch in t:
		epoch_loss = 0
		for x, y in zip(xs, ys):
			predicted = net.forward(x)
			epoch_loss += loss.loss(predicted, y)
			gradient = loss.gradient(predicted, y)
			net.backward(gradient)
			optimizer.step(net)
		accuracy = fizzbuzz_accuracy(101, 1024, net)
		t.set_description(f'fb потеря: {epoch_loss:.3f} точность: {accuracy:.2f}')

print('Тестовые результаты', fizzbuzz_accuracy(1, 101, net))

class Dropout(Layer):
	def __init__(self, p: float) -> None:
		self.p = p
		self.train = True

	def forward(self, input: Tensor) -> Tensor:
		if self.train:
			self.mask = tensor_apply(
							lambda _: 0 if random.random() < self.p else 1, input)
			return tensor_combine(operator.mul, input, self.mask)
		else:
			return tensor_apply(lambda x: x * (1 - self.p), input)

	def backward(self, gradient: Tensor) -> Tensor:
		if self.train:
			return tensor_combine(operator.mul, gradient, self.mask)
		else:
			raise RuntimeError('Не вызывайте backward в тренировочном режиме')

import mnist

mnist.temporary_dir = lambda: 'path' #Change to your path

train_images = mnist.train_images().tolist()
train_labels = mnist.train_labels().tolist()

assert shape(train_images) == [60000, 28, 28]
assert shape(train_labels) == [60000]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(10, 10)

for i in range(10):
	for j in range(10):
		ax[i][j].imshow(train_images[10 * i + j], cmap = 'Greys')
		ax[i][j].xaxis.set_visible(False)
		ax[i][j].yaxis.set_visible(False)

plt.show()

test_images = mnist.test_images().tolist()
test_labels = mnist.test_labels().tolist()

assert shape(test_images) == [10000, 28, 28]
assert shape(test_labels) == [10000]

avg = tensor_sum(train_images) / 60000 / 28 / 28

train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
				for image in train_images]
test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
			   for image in test_images]

assert -0.0001 < tensor_sum(train_images) < 0.0001

def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
	return [1.0 if j == i else 0.0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels = 5) == [0, 0, 1, 0, 0]

train_labels = [one_hot_encode(label) for label in train_labels]
test_labels = [one_hot_encode(label) for label in test_labels]

assert shape(train_labels) == [60000, 10]
assert shape(test_labels) == [10000, 10]

def loop(model: Layer, images: List[Tensor], labels: List[Tensor],
		 loss: Loss, optimizer: Optimizer = None) -> None:
	correct = 0
	total_loss = 0.0
	with tqdm.trange(len(images)) as t:
		for i in t:
			predicted = model.forward(images[i])
			if argmax(predicted) == argmax(labels[i]):
				correct += 1
			total_loss += loss.loss(predicted, labels[i])
			if optimizer is not None:
				gradient = loss.gradient(predicted, labels[i])
				model.backward(gradient)
				optimizer.step(model)
			avg_loss = total_loss / (i + 1)
			acc = correct / (i + 1)
			t.set_description(f'MNIST потеря: {avg_loss:.3f} точность: {acc:.3f}')

model = Linear(784, 10)
loss = SoftmaxCrossEntropy()
optimizer = Momentum(learning_rate = 0.01, momentum = 0.99)

loop(model, train_images, train_labels, loss, optimizer)
loop(model, test_images, test_labels, loss)

dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

model = Sequential([Linear(784, 30),
					dropout1,
					Tanh(),
					Linear(30, 10),
					dropout2,
					Tanh(),
					Linear(10, 10)])

optimizer = Momentum(learning_rate = 0.01, momentum = 0.99)
loss = SoftmaxCrossEntropy()

dropout1.train = dropout2.train = True
loop(model, train_images, train_labels, loss, optimizer)

dropout1.train = dropout2.train = False
loop(model, test_images, test_labels, loss)

import json

def save_weights(model: Layer, filename: str) -> None:
	weigths = list(model.params())
	with open(filename, 'w') as f:
		json.dum(weights, f)

def load_weigths(model: Layer, filename: str) -> None:
	with open(filename) as f:
		weights = json.load(f)
	assert all(shape(param) == shape(weight)
			   for param, weight in zip(model.params(), weights))
	for param, weight in zip(model.params(), weights):
		param[:] = weight
