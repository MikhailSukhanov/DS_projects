CREATE TABLE worst_authors
SELECT author, AVG(amount) AS author_avg, (SELECT AVG(amount) FROM book) AS general_avg
FROM book
GROUP BY author
HAVING AVG(amount) > (SELECT AVG(amount) FROM book);
SELECT * FROM worst_authors
