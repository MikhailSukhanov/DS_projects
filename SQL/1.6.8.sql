SELECT MONTHNAME(date_first) AS Месяц, COUNT(*) AS Количество
FROM trip
GROUP BY MONTHNAME(date_first)
ORDER BY COUNT(*) DESC, MONTHNAME(date_first)
