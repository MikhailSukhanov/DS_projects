SELECT city, COUNT(*) AS Количество
FROM trip
GROUP BY city
ORDER BY city
