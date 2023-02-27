from matplotlib import pyplot as plt

data = [ ('big data', 100, 15), ('Hadoop', 95, 25), ('Python', 75, 50),
         ('R', 50, 40), ('machine learning', 80, 20), ('statistics', 20, 60),
         ('data science', 60, 70), ('analytics', 90, 3),
         ('team player', 85, 85), ('dynamic', 2, 90), ('synergies', 70, 0),
         ('actionable insights', 40, 30), ('think out of the box', 45, 10),
         ('self-starter', 30, 50), ('customer focus', 65, 15),
         ('thought leadership', 35, 35)]

def text_size(total: int) -> float:
    return 8 + total / 200 * 20

for word, job_popularity, resume_popularity in data:
    plt.text(job_popularity, resume_popularity, word, ha = 'center',
             va = 'center', size = text_size(job_popularity + resume_popularity))
plt.xlabel('Популярность среди объявлений о вакансиях')
plt.ylabel('Популярность среди резюме')
plt.axis([0, 100, 0, 100])
plt.show()

def fix_unicode(text: str) -> str:
    return text.replace(u'\u2019', "'")

import re
from bs4 import BeautifulSoup
import requests

url = 'https://www.oreilly.com/ideas/what-is-data-science'
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
content = soup.find('div', 'article-body')
regex = r"[\w']+|[\.]"
document = []

from collections import defaultdict

transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
    transitions[prev].append(current)

def generate_using_bigrams() -> str:
    current = '.'
    result = []
    while True:
        next_word_candidates = transitions[current]
        current = random.choice(next_word_candidates)
        result.append(current)
        if current == '.':
            return ' '.join(result)

trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in zip(document, document[1:], document[2:]):
    if prev == '.':
        start.append(current)
    trigram_transitions[(prev, current)].append(next)

def generate_using_trigrams() -> str:
    current = random.choice(starts)
    prev = '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)
        prev, current = current, next_word
        result.append(current)
        if current == '.':
            return ' '.join(result)

from typing import List, Dict

Grammar = Dict[str, List[str]]

grammar = {
    '_S'  : ['_NP _VP'],
    '_NP' : ['_N', '_A _NP _P _A _N'],
    '_VP' : ['_V', '_V _NP'],
    '_N'  : ['data science', 'Python', 'regression'],
    '_A'  : ['big', 'linear', 'logistic'],
    '_P'  : ['about', 'near'],
    '_V'  : ['learns', 'trains', 'tests', 'is']
    }

def is_terminal(token: str) -> bool:
    return token[0] != '_'

def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        if is_terminal(token):
            continue
        replacement = random.choice(grammar[token])
        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        return expand(grammar, tokens)
    return tokens

def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ['_S'])

from typing import Tuple
import random

def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample() -> Tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_givem_x(x: int) -> int:
    return x + roll_a_die()

def random_x_given_y(y: int) -> int:
    if y <= 7:
        return random.randrange(1, y)
    else:
        return random.randrange(y - 6, 7)

def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y
def copmare_distributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

def sample_from(weights: List[float]) -> int:
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0:
            return i

from collections import Counter

draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
assert 10 < draws[0] < 190
assert 10 < draws[1] < 190
assert 650 < draws[2] < 950
assert draws[0] + draws[1] + draws[2] == 1000

documents = [
    ['Hadoop', 'Big Data', 'HBase', 'Java', 'Spark', 'Storm', 'Cassandra'],
    ['NoSQL', 'MongoDB', 'Cassandra', 'HBase', 'Postgres'],
    ['Python', 'scikit-learn', 'scipy', 'numpy', 'statsmodels', 'pandas'],
    ['R', 'Python', 'statistics', 'regression', 'probability'],
    ['machine learning', 'regression', 'decision trees', 'libsvm'],
    ['Python', 'R', 'Java', 'C++', 'Haskell', 'programming languages'],
    ['statistics', 'probability', 'mathematics', 'theory'],
    ['machine learning', 'scikit-learn', 'Mahout', 'neural networks'],
    ['neural networks', 'deep learning', 'Big Data', 'artificial intelligence'],
    ['Hadoop', 'Java', 'MapReduce', 'Big Data'],
    ['statistics', 'R', 'statsmodels'],
    ['C++', 'deep learning', 'artificial intelligence', 'probability'],
    ['pandas', 'R', 'Python'],
    ['databases', 'HBase', 'Postgres', 'MySQL', 'MongoDB'],
    ['libsvm', 'regression', 'support vector machines']]

K = 4

document_topic_counts = [Counter() for _ in documents]
topic_word_counts = [Counter() for _ in range(K)]
topic_counts = [0 for _ in range(K)]
document_lengths = [len(document) for document in documents]
distinct_words = set(word for document in documents for word in document)

W = len(distinct_words)
D = len(documents)

def p_topic_given_document(topic: int, d: int,
                           alpha: float = 0.1) -> float:
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word: str, topic: int,
                       beta: float = 0.1) -> float:
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))

def topic_weight(d: int, word: str, k: int) -> float:
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k) for k in range(K)])

random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

import tqdm

for iter in tqdm.trange(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0:
            print(k, word, count)

import math

Vector = List[int]

def dot(v: Vector, w: Vector) -> float:
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))

assert cosine_similarity([1, 1, 1], [2, 2, 2]) == 1
assert cosine_similarity([-1, -1], [2, 2]) == -1
assert cosine_similarity([1, 0], [0, 1]) == 0

colors = ['red', 'green', 'blue', 'yellow', 'black', '']
nouns = ['bed', 'car', 'boat', 'cat']
verbs = ['is', 'was', 'seems']
adverbs = ['very', 'quite', 'extremely', '']
adjectives = ['slow', 'fast', 'soft', 'hard']

def make_sentence() -> str:
    return ' '.join(['The',
                     random.choice(colors),
                     random.choice(nouns),
                     random.choice(verbs),
                     random.choice(adverbs),
                     random.choice(adjectives),
                     '.'])

NUM_SENTENCES = 50
sentences = [make_sentence() for _ in range(NUM_SENTENCES)]

Tensor = list

class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}

        for word in (words or []):
            self.add(word)

    @property
    def size(self) -> int:
        return len(self.w2i)

    def add(self, word: str) -> None:
        if word not in self.w2i:
            word_id = len(self.w2i)
            self.w2i[word] = word_id
            self.i2w[word_id] = word

    def get_id(self, word: str) -> int:
        return self.w2i.get(word)

    def get_word(self, word_id: int) -> str:
        return self.i2w.get(word_id)

    def one_hot_encode(self, word: str) -> Tensor:
        word_id = self.get_id(word)
        assert word_id is not None, f'Неизвестное слово {word}'
        return [1 if i == word_id else 0 for i in range(self.size)]

vocab = Vocabulary(['a', 'b', 'c'])
assert vocab.size == 3
assert vocab.get_id('b') == 1
assert vocab.one_hot_encode('b') == [0, 1, 0]
assert vocab.get_id('z') is None
assert vocab.get_word(2) == 'c'

vocab.add('z')
assert vocab.size == 4
assert vocab.get_id('z') == 3
assert vocab.one_hot_encode('z') == [0, 0, 0, 1]

import json

def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(vocab.w2i, f)

def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename) as f:
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
    return vocab

from typing import Iterable, Callable

class Layer:
    def forward(self, imput):
        raise NotImplementError

    def backward(self, gradient):
        raise NotImplementError

    def params(self) -> Iterable[Tensor]:
        return ()

    def grads(self) -> Iterable[Tensor]:
        return ()

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

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

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

def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean = mean, variance = variance)
                for _ in range(dims[0])]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]

def is_1d(tensor: Tensor) -> bool:
    return not isinstance(tensor[0], list)

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)
        self.last_input_id = None

    def forward(self, input_id: int) -> Tensor:
        self.input_id = input_id
        return self.embeddings[input_id]

    def backward(self, gradient: Tensor) -> None:
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row
        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient

    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]

    def grads(self) -> Iterable[Tensor]:
        return [self.grad]

class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        super().__init__(vocab.size, embedding_dim)
        self.vocab = vocab

    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None

    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        vector = self[word]
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse = True)
        return scores[:n]

tokenized_sentences = [re.findall('[a-z]+|[.]', sentence.lower())
                       for sentence in sentences]

vocab = Vocabulary(word
                   for sentence_words in tokenized_sentences
                   for word in sentence_words)

def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]

inputs: List[int] =[]
targets: List[Tensor] = []

for sentence in tokenized_sentences:
    for i, word in enumerate(sentence):
        for j in [i - 2, i - 1, i + 1, i + 2]:
            if 0 <= j < len(sentence):
                nearby_word = sentence[j]
                inputs.append(vocab.get_id(word))
                targets.append(vocab.one_hot_encode(nearby_word))

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

EMBEDDING_DIM = 5
embedding = TextEmbedding(vocab = vocab, embedding_dim = EMBEDDING_DIM)

model = Sequential([embedding, Linear(input_dim = EMBEDDING_DIM, output_dim = vocab.size)])

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

def softmax(tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor]

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor, t2: Tensor) -> Tensor:
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

def tensor_sum(tensor: Tensor) -> float:
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)

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

class Optimizer:
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

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
class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = tensor_combine(
                        lambda param, grad: param - grad * self.lr,
                        param, grad)

loss = SoftmaxCrossEntropy()
optimizer = GradientDescent(learning_rate = 0.01)

for epoch in range(100):
    epoch_loss = 0.0
    for input, target in zip(inputs, targets):
        predicted = model.forward(input)
        epoch_loss += loss.loss(predicted, target)
        gradient = loss.gradient(predicted, target)
        model.backward(gradient)
        optimizer.step(model)
    print(epoch, epoch_loss)
    print(embedding.closest('black'))
    print(embedding.closest('slow'))
    print(embedding.closest('car'))

pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
         for w1 in vocab.w2i
         for w2 in vocab.w2i
         if w1 < w2]

pairs.sort(reverse = True)
print(pairs[:5])

def tanh(x: float) -> float:
    if x < -100: return -1
    elif x > 100: return 1
    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)

class SimpleRnn(Layer):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w = random_tensor(hidden_dim, input_dim, init = 'xavier')
        self.u = random_tensor(hidden_dim, hidden_dim, init = 'xavier')
        self.b = random_tensor(hidden_dim)
        self.reset_hidden_state()

    def reset_hidden_state(Self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        self.prev_hidden = self.hidden
        a = [(dot(self.w[h], input) + dot(self.u[h], self.hidden), self.b[h])
             for h in range(self.hidden_dim)]
        self.hidden = tensor_apply(tanh, a)
        return self.hidden

    def backward(self, gradient: Tensor):
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]
        self.b_grad = a_grad
        self.w_grad = [[a_grad[h] * self.input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]
        return [sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]
