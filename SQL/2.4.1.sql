SELECT buy.buy_id, title, price, buy_book.amount
FROM buy
     INNER JOIN buy_book ON buy.buy_id = buy_book.buy_id
     INNER JOIN book ON book.book_id = buy_book.book_id
WHERE client_id = (SELECT client_id
                   FROM client
                   WHERE name_client = 'Баранов Павел')
ORDER BY buy.buy_id, title
