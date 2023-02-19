with open('email_addresses.txt', 'w') as f:
    f.write('john@gmail.com\n')
    f.write('john@m.datasciencester.com\n')
    f.write('johnsmith@m.datasciencester.com\n')

def get_domain(email_adress: str) -> str:
	return email_adress.lower().split('@')[-1]

assert get_domain('john@gmail.com') == 'gmail.com'
assert get_domain('john@m.datasciencester.com') == 'm.datasciencester.com'

from collections import Counter

with open('email_addresses.txt', 'r') as f:
	domain_counts = Counter(get_domain(line.strip())
							for line in f
							if '@' in line)

with open('tab_delimited_stock_prices.txt', 'w') as f:
    f.write('''6/20/2014\tAAPL\t90.91
6/20/2014\tMSFT\t41.68
6/20/2014\tFB\t64.5
6/19/2014\tAAPL\t91.86
6/19/2014\tMSFT\t41.51
6/19/2014\tFB\t64.34
''')

import csv

with open('tab_delimited_stock_prices.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])

with open('colon_delimited_stock_prices.txt', 'w') as f:
    f.write('''date:symbol:closing_price
6/20/2014:AAPL:90.91
6/20/2014:MSFT:41.68
6/20/2014:FB:64.5
''')

with open('colon_delimited_stock_prices.txt') as f:
	colon_reader = csv.DictReader(f, delimiter = ':')
	for row in colon_reader:
		date = row['date']
		symbol = row['symbol']
		closing_price = float(row['closing_price'])

today_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('comma_delimited_stock_prices.txt', 'w') as f:
	writer = csv.writer(f, delimiter = ',')
	for stock, price in today_prices.items():
		writer.writerow([stock, price])

from bs4 import BeautifulSoup
import requests

url = 'https://www.washingtonpost.com/technology/2023/02/03/nuclear-rocket-darpa-nasa/'
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph = soup.find('p')
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

first_paragraph_id = first_paragraph.get('id')
all_paragraphs = soup.find_all('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

important_paragraphs1 = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
						 if 'important' in p.get('class', [])]

spans_inside_divs = [span
					 for div in soup('div')
					 for span in div('span')]

url = 'https://www.house.gov/representatives'
text = requests.get(url).text
soup = BeautifulSoup(text, 'html5lib')

all_urls = [a['href'] for a in soup('a') if a.has_attr('href')]
print(len(all_urls))

import re

regex = r'^https?://.*\.house\.gov/?$'
good_urls = [url for url in all_urls if re.match(regex, url)]
print(len(good_urls))

good_urls = list(set(good_urls))
print(len(good_urls))

html = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
print(links)

from typing import Dict, Set

press_releases: Dict[str, Set[str]] = {}
for house_url in good_urls:
	html = requests.get(house_url).text
	soup = BeautifulSoup(html, 'html5lib')
	pr_links = {a['href'] for a in soup('a')
				if 'press releases' in a.text.lower()}
	print(f'{house_url}: {pr_links}')
	press_releases[house_url] = pr_links

def paragraph_mentions(text: str, keyword: str) -> bool:
	soup = BeautifulSoup(text, 'html5lib')
	paragraphs = [p.get_text() for p in soup('p')]
	return any(keyword.lower() in paragraph.lower()
			   for paragraph in paragraphs)

text = '''<body><h1>Facebook</h1><p>Twitter</p>'''
assert paragraph_mentions(text, 'twitter')
assert not paragraph_mentions(text, 'facebook')

for house_url, pr_links in press_releases.items():
	for pr_link in pr_links:
		url = f'{house_url}/{pr_link}'
		text = requests.get(url).text
		if paragraph_mentions(text, 'data'):
			print(f'{house_url}')
			break

import json

serialized = '''{ "title" : "Data Science Book",
                  "author" : "Bob Smith",
                  "publicationYear" : 2015,
                  "topics" : [ "data", "science", "data science"] }'''

deserialized = json.loads(serialized)
assert deserialized['publicationYear'] == 2015
assert 'data science' in deserialized['topics']

github_user = 'MikhailSukhanov'
endpoint = f'https://api.github.com/users/{github_user}/repos'
repos = json.loads(requests.get(endpoint).text)

from dateutil.parser import parse

dates = [parse(repo['created_at']) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

last_repositories = sorted(repos, key = lambda r: r['created_at'],
						   reverse = True)[:5]
last_languages = [repo['language'] for repo in last_repositories]
