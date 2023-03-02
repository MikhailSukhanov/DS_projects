DELETE FROM attempt
WHERE DATEDIFF(date_attempt, "2020.05.01") < 0;
SELECT * FROM attempt;
SELECT * FROM testing
