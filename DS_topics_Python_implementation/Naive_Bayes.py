from typing import Set
import re

def tokenize(text: str) -> Set[str]:
	text = text.lower()
	all_words = re.findall('[a-z0-9]+', text)
	return set(all_words)

assert tokenize('Data Science is science') == {'data', 'science', 'is'}

from typing import NamedTuple

class Message(NamedTuple):
	text: str
	is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
	def __init__(self, k: float = 0.5) -> None:
		self.k = k
		self.tokens: Set[str] = set()
		self.token_spam_counts: Dict[str, int] = defaultdict(int)
		self.token_ham_counts: Dict[str, int] = defaultdict(int)
		self.spam_messages = self.ham_messages = 0

	def train(self, messages: Iterable[Message]) -> None:
		for message in messages:
			if message.is_spam:
				self.spam_messages += 1
			else:
				self.ham_messages += 1
			for token in tokenize(message.text):
				self.tokens.add(token)
				if message.is_spam:
					self.token_spam_counts[token] += 1
				else:
					self.token_ham_counts[token] += 1

	def _probabilities(self, token: str) -> Tuple[float, float]:
		spam = self.token_spam_counts[token]
		ham = self.token_ham_counts[token]
		p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
		p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
		return p_token_spam, p_token_ham

	def predict(self, text: str) -> float:
		text_tokens = tokenize(text)
		log_prob_if_spam = log_prob_if_ham = 0.0
		for token in self.tokens:
			prob_if_spam, prob_if_ham = self._probabilities(token)
			if token in text_tokens:
				log_prob_if_spam += math.log(prob_if_spam)
				log_prob_if_ham += math.log(prob_if_ham)
			else:
				log_prob_if_spam += math.log(1.0 - prob_if_spam)
				log_prob_if_ham += math.log(1.0 - prob_if_ham)
		prob_if_spam = math.exp(log_prob_if_spam)
		prob_if_ham = math.exp(log_prob_if_ham)
		return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message('spam rules', is_spam = True),
        	Message('ham rules', is_spam = False),
        	Message('hello ham', is_spam = False)]

model = NaiveBayesClassifier(k = 0.5)
model.train(messages)

assert model.tokens == {'spam', 'ham', 'rules', 'hello'}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {'spam': 1, 'rules': 1}
assert model.token_ham_counts == {'ham': 2, 'rules': 1, 'hello': 1}
