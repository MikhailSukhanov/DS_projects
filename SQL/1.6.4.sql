SELECT city, COUNT(*) AS Количество
FROM trip
GROUP BY city
ORDER BY COUNT(*) DESC
LIMIT 2
