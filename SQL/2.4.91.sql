SELECT title, SUM(Количество) AS Количество, SUM(Сумма) AS Сумма
FROM (
    SELECT title, SUM(buy_book.amount) AS Количество, SUM(price * buy_book.amount) AS Сумма
    FROM book
         INNER JOIN buy_book USING(book_id)
         INNER JOIN buy ON buy_book.buy_id = buy.buy_id
         INNER JOIN buy_step ON buy.buy_id = buy_step.buy_id
    WHERE step_id = 1 AND date_step_end IS NOT NULL
    GROUP BY title
    UNION ALL
    SELECT title, SUM(buy_archive.amount) AS Количество, SUM(buy_archive.price * buy_archive.amount) AS Сумма
    FROM buy_archive
         INNER JOIN book ON buy_archive.book_id = book.book_id
    GROUP BY title) AS first_01
GROUP BY title
ORDER BY SUM(Сумма) DESC
