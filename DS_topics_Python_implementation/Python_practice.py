for i in [1, 2, 3, 4, 5]:
	print(i)
	for j in [1, 2, 3, 4, 5]:
		print(j)
		print(i + j)
	print(i)
print('The end')

long_winded_computation = (1 + 2 + 3 + 4 + 5)
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
easier_to_read_list_of_lists = [[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9]]
two_plus_three = 2 + \
		3

import re as regex
my_regex = regex.compile('[0-9]+', regex.I)

from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

def double(x):
	return x * 2

def apply_to_one(f):
	return f(1)

my_double = double
x = apply_to_one(my_double)
y = apply_to_one(lambda x: x + 4)

def my_print(message = 'My default message'):
	print(message)

my_print('Hello!')
my_print()

def full_name(first = 'некто', last = 'как-то там'):
	return first + ' ' + last

full_name('A', 'B')
full_name('A')
full_name(last = 'B')

tab_string = '\t'
len(tab_string)
not_tab_string = r'\t'
len(not_tab_string)

multi_line_string = '''First string.
Second string.
Third string.'''

first_name = 'Adam'
last_name = 'Floyd'
full_name1 = first_name + ' ' + last_name
full_name2 = '{0} {1}'.format(first_name, last_name)
full_name3 = f'{first_name} {last_name}'

try:
	print(0 / 0)
except ZeroDivisionError:
	print('You can not divide by zero')

integer_list = [1, 2, 3]
heterogeneous_list = ['string', 0.1, True]
list_of_listst = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)
list_sum = sum(integer_list)

x = list(range(10))
zero = x[0]
nine = x[-1]
x[0] = -1
first_three = x[:3]
last_three = x[-3:]
copy_of_x = x[:]
every_third = x[::3]
five_to_three = x[5:2:-1]

1 in [1, 2, 3]
0 in [1, 2, 3]

x = [1, 2 , 3]
x.extend([4, 5, 6])

x = [1, 2, 3]
y = x + [4, 5, 6]

x = [1, 2, 3]
x.append(0)
y = x[-1]
z = len(x)

x, y = [1, 2]
_, y = [1, 2]

my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3

try:
	my_tuple[1] = 3
except TypeError:
	print('Кортеж нельзя модифицировать')

def sum_and_product(x, y):
	return (x + y), (x * y)

sp = sum_and_product(2, 3)
s, p = sum_and_product(5, 10)

x, y = 1, 2
x, y = y, x

empty_dict1 = {}
empty_dict2 = dict()
grades = {'Bob': 80, 'Dan': 95}
bobs_grade = grades['Bob']

try:
	kates_grade = grades['Kate']
except KeyError:
	print('Оценки для Кейт отстутствуют!')

bob_has_grade = 'Bob' in grades
kate_has_grade = 'Kate' in grades

bobs_grade = grades.get('Bob', 0)
kates_grade = grades.get('Kate', 0)
no_ones_grade = grades.get('Никто')

grades['Dan'] = 99
grades['Kate'] = 100
num_students = len(grades)

tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}
tweet_keys = tweet.keys()
tweet_values = tweet.values()
tweet_items = tweet.items()

'user' in tweet_keys
'user' in tweet
'joelgrus' in tweet_values

word_counts = {}
document = {}
for word in document:
	if word in word_counts:
		word_counts[word] += 1
	else:
		word_counts[word] = 1

word_counts = {}
for word in document:
	try:
		word_counts[word] += 1
	except KeyError:
		word_counts[word] = 1

word_counts = {}
for word in document:
	previous_count = word_counts.get(word, 0)
	word_counts[word] = previous_count + 1

from collections import defaultdict

word_counts = defaultdict(int)
for word in document:
	word_count[word] += 1

dd_list = defaultdict(list)
dd_list[2].append(1)

dd_dict = defaultdict(dict)
dd_dict['Джоел']['город'] = 'Сиэтл'

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1

from collections import Counter
c = Counter([0, 1, 2, 0])
word_counts = Counter(document)

for word, count in word_counts.most_common(10):
	print(word, count)

s = set()
s.add(1)
s.add(2)
s.add(2)
x = len(s)
y = 2 in s
z = 3 in s

hundreds_of_other_words = []
stopwords_list = ['a', 'an', 'at'] + hundreds_of_other_words + ['yet', 'you']
'zip' in stopwords_list

stopwords_set = set(stopwords_list)
'zip' in stopwords_set

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)
item_set = set(item_list)
num_distinct_items = len(item_set)
distinct_item_list = list(item_set)

if 1 > 2:
	message = 'A'
elif 1 > 3:
	message = 'B'
else:
	message = 'C'

parity = 'четное' if x % 2 == 0 else 'нечетное'

x = 0
while x < 10:
	print(x, 'меньше 10')
	x += 1

for x in range(10):
	if x == 3:
		continue
	if x == 5:
		break
	print(x)

one_is_less_than_two = 1 < 2
true_equals_false = True == False

x = None
assert x is None

def some_function_that_returns_a_string():
    return ""

s = some_function_that_returns_a_string()
if s:
	first_char = s[0]
else:
	first_char = ''
first_char = s and s[0]

safe_x = x or 0

all([True, 1, {3}])
all([True, 1, {}])
any([True, 1, {}])
all([])
any([])

x = [4, 1, 2, 3]
y = sorted(x)
x.sort()

x = sorted([-4, 1, -2, 3], key = abs, reverse = True)
wc = sorted(word_counts.items(), key = lambda word_and_count: word_and_count[1],
			reverse = True)

even_numbers = [x for x in range(5) if x % 2 == 0]
squares = [x * x for x in range(5)]
even_squares = [x * x for x in even_numbers]

square_dict = {x: x * x for x in range(5)}
square_set = {x * x for x in [1, -1]}

zeros = [0 for _ in even_numbers]
pairs = [(x, y)
		 for x in range(10)
		 for y in range(10)]
increasing_pairs = [(x, y)
					for x in range(10)
					for y in range(x + 1, 10)]

assert 1 + 1 == 2
assert 1 + 1 == 2, '1 + 1 должно равняться двум, но здесь это не так'

def smallest_item(xs):
	return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2])	== -1

def smallest_item(xs):
	assert xs, 'пустой список не имеет наименьшего элемента'
	return min(xs)

class CountingClicker:
	def __init__(self, count = 0):
		self.count = count

	def __repr__(self):
		return f'CountingClicker(count = {self.count})'

	def click(self, num_times = 1):
		self.count += num_times

	def read(self):
		return self.count

	def reset(self):
		self.count = 0

clicker1 = CountingClicker()
clicker2 = CountingClicker(100)
clicker3 = CountingClicker(count = 100)

clicker = CountingClicker()
assert clicker.read() == 0, 'счетчик должен начинаться со значения 0'
clicker.click()
clicker.click()
assert clicker.read() == 2, 'после двух нажатий счетчик должен иметь значение 2'
clicker.reset()
assert clicker.read() == 0, 'после сброса счетчик должен вернуться к 0'

class NoResetClicker(CountingClicker):
	def reset(self):
		pass

clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, 'функция reset не должна ничего делать'

def generate_range(n):
	i = 0
	while i < n:
		yield i
		i += 1
for i in generate_range(10):
	print(f'i: {i}')

def natural_numbers():
	n = 1
	while True:
		yield n
		n += 1

evens_below_20 = (i for i in generate_range(20) if i % 2 == 0)

data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)

names = ['Alice', 'Bob', 'Charlie', 'Debbie']
for i, name in enumerate(names):
	print(f'name {i} is {name}')

import random
random.seed(10)

four_uniform_randoms = [random.random() for _ in range(4)]

random.randrange(10)
random.randrange(3, 6)

up_to_ten = list(range(10))
random.shuffle(up_to_ten)

my_best_frtiend = random.choice(['Alice', 'Bob', 'Charlie'])

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)

four_with_replacement = [random.choice(range(10)) for _ in range(4)]

import re

re_examples = [not re.match('a', 'cat'),
			   re.search('a', 'cat'),
			   not re.search('c', 'dog'),
			   3 == len(re.split('[ab]', 'carbs')),
			   'R-D-' == re.sub('[0-9]', '-', 'R2D2')]

assert all(re_examples)

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
[pair for pair in zip(list1, list2)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

def add(a, b): return a + b

add(1, 2)

try:
	add([1, 2])
except TypeError:
	print('Функция add ожидает два входа')

add(*[1, 2])

def doubler(f):
	def g(x):
		return 2 * f(x)
	return g

def f1(x):
	return x + 1

g = doubler(f1)
assert g(3) == 8, '(3 + 1) * 2 должно быть равно 8'
assert g(-1) == 0, '(-1 + 1) * 2 должно быть равно 0'

def f2(x, y):
	return x + y

def magic(*args, **kwargs):
	print('Безымянные аргументы:', args)
	print('Аргументы по ключу:', kwargs)

magic(1, 2, key1 = 'word1', key2 = 'word2')

def other_way_magic(x, y, z):
	return x + y + z

x_y_list = [1, 2]
z_dict = {'z': 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, '1 + 2 + 3 должно быть равно 6'

def doubler_correct(f):
	def g(*args, **kwargs):
		return 2 * f(*args, **kwargs)
	return g

g = doubler_correct(f2)
print(g(1, 2))

def add(a, b):
	return a + b

assert add(10, 5) == 15, '+ является допустимым для чисел'
assert add([1, 2], [3]) == [1, 2, 3], '+ является допустимым для списков'
assert add('Эй, ', 'привет!') == 'Эй, привет!', '+ является допустимым для строк'

try:
	add(10, 'пять')
except TypeError:
	print('Невозможно прибавить целое число к строке')

def add(a: int, b: int) -> int:
	return a + b

add(10, 5)
add('Эй, ', 'там')

def total(xs: list) -> float:
	return sum(xs)

from typing import List

def total(xs: List[float]) -> float:
	return sum(xs)

x: int = 5

values = []
best_so_far = None

from typing import Optional

values: List[int] = []
best_so_far: Optional[float] = None

from typing import Dict, Iterable, Tuple

lazy = True

counts: Dict[str, int] = {'data': 1, 'science': 2}

if lazy:
	evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
else:
	evens = [0, 2, 4, 6, 8]

triple: Tuple[int, float, int] = (10, 2.3, 5)

from typing import Callable

def comma_repeater(s: str, n: int) -> str:
	n_copies = [s for _ in range(n)]
	return ', '.join(n_copies)

def twice(repeater: Callable[[str, int], str], s: str) -> str:
	return repeater(s, 2)

assert twice(comma_repeater, 'подсказки типов') == 'подсказки типов, подсказки типов'

Number = int
Numbers = List[Number]

def total(xs: Numbers) -> Number:
	return sum(xs)
