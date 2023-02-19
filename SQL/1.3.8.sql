SELECT author, MAX(price) AS Максимальная_цена, SUM(amount) AS Общее_количество
FROM book
GROUP BY author
HAVING SUM(amount) > 10
ORDER BY Максимальная_цена DESC;
