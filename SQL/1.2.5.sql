SELECT title, author, amount,
    ROUND(price * (100-30)/100, 2) AS new_price
FROM book;
