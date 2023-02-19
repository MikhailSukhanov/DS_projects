SELECT title, author, price, amount
FROM book
WHERE (price < 500 OR price > 600) AND amount * price >= 5000;
