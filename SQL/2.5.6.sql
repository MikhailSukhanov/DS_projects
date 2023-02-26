CREATE TABLE buy_pay AS
SELECT buy_id, SUM(buy_book.amount) AS Количество, SUM(price * buy_book.amount) AS Итого
FROM buy_book
     INNER JOIN book USING (book_id)
WHERE buy_id = 5
GROUP BY buy_id;
SELECT * FROM buy_pay
