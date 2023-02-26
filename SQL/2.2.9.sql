SELECT *
FROM genre
     LEFT JOIN book ON genre.genre_id = book.genre_id
     INNER JOIN author USING(author_id)
