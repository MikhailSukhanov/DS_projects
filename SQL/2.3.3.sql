INSERT INTO book (title, author_id, price, amount)
SELECT title, author_id, price, amount
FROM supply
     INNER JOIN author ON supply.author = author.name_author
WHERE amount <> 0;
SELECT * FROM book
