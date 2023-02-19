from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

plt.plot(years, gdp, color = 'green', marker = 'o', linestyle = 'solid')
plt.title('Номинальный ВВП')
plt.ylabel('Млрд $')
plt.show()

movies = ['Трансформеры 1', 'Форсаж 5', 'Пираты Карибского моря 2', 'Джокер', 'Интерстеллар']
num_oscars = [3, 6, 4, 5, 9]

plt.bar(range(len(movies)), num_oscars)
plt.title('Мои любимые фильмы')
plt.ylabel('Количество наград')
plt.xticks(range(len(movies)), movies)
plt.show()

from collections import Counter

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()], list(histogram.values()), 10)
plt.axis([-5, 105, 0, 5])
plt.xticks([10 * i for i in range(11)])
plt.xlabel('Дециль')
plt.ylabel('Число студентов')
plt.title('Распределение оценок за экзамен №1')
plt.show()

mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel('Число упоминания науки о данных')
plt.ticklabel_format(useOffset = False)
plt.axis([2016.5, 2018.5, 499, 506])
plt.title('Введение в заблуждение относительно прироста')
plt.show()

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel('Число упоминания науки о данных')
plt.ticklabel_format(useOffset = False)
plt.axis([2016.5, 2018.5, 0, 530])
plt.title('Правильный график')
plt.show()

variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

plt.plot(xs, variance, 'g-', label = 'Дисперсия')
plt.plot(xs, bias_squared, 'r-.', label = 'Смещение^2')
plt.plot(xs, total_error, 'b:', label = 'Суммарная ошибка')
plt.legend(loc = 9)
plt.xlabel('Сложность модели')
plt.title('Компромисс между смещением и дисперсией')
plt.show()

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

for label, friend_count, minute_count in zip(labels, friends, minutes):
	plt.annotate(label, xy = (friend_count, minute_count), xytext = (5, -5),
				 textcoords = 'offset points')

plt.title('Число минут против числа друзей')
plt.xlabel('Число друзей')
plt.ylabel('Число минут, проводимых на сайте ежедневно')
plt.show()

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title('Оси не сопоставимы')
plt.xlabel('Оценки за экзамен 1')
plt.ylabel('Оценки за экзамен 2')
plt.show()

plt.scatter(test_1_grades, test_2_grades)
plt.title('Оси сопоставимы')
plt.xlabel('Оценки за экзамен 1')
plt.ylabel('Оценки за экзамен 2')
plt.axis('equal')
plt.show()
