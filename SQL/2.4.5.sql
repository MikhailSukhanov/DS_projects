SELECT buy.buy_id, name_client, SUM(price * buy_book.amount) AS Стоимость
FROM client
     INNER JOIN buy ON client.client_id = buy.client_id
     INNER JOIN buy_book ON buy.buy_id = buy_book.buy_id
     INNER JOIN book ON book.book_id = buy_book.book_id
GROUP BY buy.buy_id
ORDER BY buy.buy_id
