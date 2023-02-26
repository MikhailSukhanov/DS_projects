SELECT book.title AS Название, author AS Автор, (supply.amount + book.amount) AS Количество
FROM author
     INNER JOIN book USING(author_id)
     INNER JOIN supply 
     ON author.name_author = supply.author AND book.title = supply.title AND book.price = supply.price
