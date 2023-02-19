UPDATE book
SET price = IF(buy = 0, price * 0.9, price), buy = IF((amount - buy) < 0, amount, buy);
SELECT * FROM book
