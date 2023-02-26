SELECT name_author
FROM author 
     INNER JOIN book ON author.author_id = book.author_id
     INNER JOIN genre ON genre.genre_id = book.genre_id
GROUP BY name_author
HAVING COUNT(DISTINCT(name_genre)) = 1
