SELECT title, author, price, ROUND((price - (SELECT AVG(price) FROM book)), 2) AS Дороже_дешевле
FROM book
WHERE price > ANY(SELECT price * 1.1 FROM book)
