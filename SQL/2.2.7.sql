SELECT title, name_author, name_genre, price, amount
FROM author
     INNER JOIN book ON author.author_id = book.author_id
     INNER JOIN genre ON genre.genre_id = book.genre_id
WHERE name_genre IN (SELECT first_1.name_genre
                     FROM (
                           SELECT name_genre, SUM(amount) AS sum_amount
                           FROM genre g
                                INNER JOIN book b ON g.genre_id = b.genre_id
                           GROUP BY name_genre) first_1
                     INNER JOIN (
                           SELECT name_genre, SUM(amount) AS sum_amount
                           FROM genre g
                                INNER JOIN book b ON g.genre_id = b.genre_id
                           GROUP BY name_genre
                           ORDER BY SUM(amount) DESC
                           LIMIT 1) second_2
                     ON first_1.sum_amount = second_2.sum_amount)
ORDER BY title
