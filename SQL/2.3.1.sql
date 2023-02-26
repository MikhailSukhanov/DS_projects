UPDATE book
       INNER JOIN author ON book.author_id = author.author_id
       INNER JOIN supply ON book.title = supply.title AND supply.author = author.name_author
SET book.amount = book.amount + supply.amount, supply.amount = 0, book.price = (book.price * book.amount + supply.price * supply.amount)/(book.amount + supply.amount)
WHERE book.price <> supply.price;
SELECT * FROM book;
SELECT * FROM supply
