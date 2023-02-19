SELECT title, ROUND(IF(author LIKE 'Б%', price * 1.2, price * 0.8), 2) AS new_price
FROM book
WHERE ROUND(IF(author LIKE 'Б%', price * 1.2, price * 0.8), 2) > 550
ORDER BY title DESC;
