SELECT name_client
FROM author
     INNER JOIN book USING(author_id)
     INNER JOIN buy_book USING(book_id)
     INNER JOIN buy USING(buy_id)
     INNER JOIN client USING(client_id)
WHERE name_author = "Достоевский Ф.М."
GROUP BY name_client
ORDER BY name_client
