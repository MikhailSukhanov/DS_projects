from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from telethon.tl.functions.messages import GetHistoryRequest
import csv
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.probability import FreqDist

api_id = 11111111 #Changed
api_hash = '111111111111111111111111111111' #Changed
phone = '11111111111' #Changed

client = TelegramClient(phone, api_id, api_hash)
client.start()

chats = []
last_date = None
size_chats = 15
groups = []

result = client(GetDialogsRequest(offset_date = last_date,
								 offset_id = 0,
								 offset_peer = InputPeerEmpty(),
								 limit = size_chats,
								 hash = 0))
chats.extend(result.chats)

for chat in chats:
	try:
		if chat.megagroup == True:
			groups.append(chat)
	except:
		continue

for g in groups:
	print(g.title)

data = []
offset_id = 0
limit = 100
total_messages = 0
total_count_limit = 0

while True:
	history = client(GetHistoryRequest(peer = groups[1],	#Aimylogic (сообщество пользователей)
								   	   offset_id = offset_id,
									   offset_date = None,
									   add_offset = 0,
									   limit = limit,
									   max_id = 0,
									   min_id = 0,
									   hash = 0))
	if not history.messages:
		break
	messages = history.messages
	for i, message in enumerate(messages):
		data.append([message.to_dict()['date'], message.message])
	offset_id = messages[len(messages) - 1].id
	if total_count_limit != 0 and total_messages >= total_count_limit:
		break

with open('chats.csv', 'w', encoding = 'UTF-8') as f:
	writer = csv.writer(f, delimiter = ',', lineterminator = '\n')
	writer.writerow(['date', 'message'])
	for d in data:
		writer.writerow(d)

#Data preprocessing
dataframe = pd.read_csv('chats.csv')
df = dataframe.copy()
df = df.dropna()
df['year'] = [i[:4] for i in df['date']]
df['month'] = [i[5:7] for i in df['date']]
df['year'] = df['year'].astype('int')
df['month'] = df['month'].astype('int')
df = df[df['year'] != 2023]
df = df.drop(columns = 'date')
df['message'] = df['message'].str.lower()
df.index = range(len(df))
df1 = df.copy()

#Removing punctuation and non-cyrillic characters
spec_chars = string.punctuation
ru_chars = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

for i in range(len(df)):
    df.iloc[i, 0] = ''.join([i for i in df.iloc[i, 0] if i not in spec_chars and i in ru_chars])

#A dataframe containing the words and the year each word was sent
w = []
y = []
w_list = []

for i in range(len(df)):
    w_list = df['message'][i].split()
    for word in w_list:
        w.append(word)
        y.append(df['year'][i])
        
words = pd.DataFrame({'word': w, 'year': y})

#Removing stop words
stopwords = stopwords.words('russian')
stopwords.append('это')
drop = []

for i in range(len(words)):
    if words['word'][i] in stopwords:
        drop.append(i)
words = words.drop(drop)

words['word'] = [i.replace('бота', 'бот') for i in words['word']]
words.index = range(len(words))

#Most used words over 5 years
years = [2018, 2019, 2020, 2021, 2022]
freq_words = pd.DataFrame(columns = years, index = range(10))

for year in years:
    fdist = FreqDist(words.loc[words['year'] == year, 'word']).most_common(10)
    freq_words[year] = [i[0] for i in fdist]
print(freq_words, '\n')

#Frequency of questions over 5 years
rel_questions = {}
for year in years:
    df_temp = df1.loc[df1['year'] == year, 'message']
    count = 0
    for m in df_temp:
        if '?' in m:
            count += 1
    rel_questions[str(year)] = round(count / len(df_temp) * 100, 2)
print(rel_questions)

#Проведя анализ фрейма данных 'freq_words' можно сделать следующие выводы:
#главным предметом обсуждения на протяжении пяти лет является 'бот';
#общение в чате происходит преимущественно в рабочее время (на это указывает присутствие
#в числе популярных слова 'день');
#в 2018 году популярными предметами обсуждения также были 'сценарий', 'блок' и 'запрос', но в
#остальные годы эти темы не были так популярны;
#
#Проведя анализ словаря 'rel_questions' можно сделать следующие выводы:
#в среднем каждое четвертое сообщение является вопросом;
#процент сообщений-вопросов меняется от года к году в среднем на ±1.35 п.п. без
#ярко выраженной тенденции
