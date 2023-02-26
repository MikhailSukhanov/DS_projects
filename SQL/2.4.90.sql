SELECT YEAR(date_payment) AS Год, MONTHNAME(date_payment) AS Месяц, SUM(price * amount) AS Сумма
FROM buy_archive
GROUP BY YEAR(date_payment), MONTHNAME(date_payment)
UNION ALL
SELECT YEAR(date_step_end) AS Год, MONTHNAME(buy_step.date_step_end) AS Месяц, SUM(price * buy_book.amount)
FROM book
     INNER JOIN buy_book ON book.book_id = buy_book.book_id
     INNER JOIN buy ON buy_book.buy_id = buy.buy_id
     INNER JOIN buy_step ON buy.buy_id = buy_step.buy_id
WHERE step_id = 1 AND date_step_end IS NOT NULL
GROUP BY YEAR(date_step_end), MONTHNAME(date_step_end)
ORDER BY Месяц, Год
