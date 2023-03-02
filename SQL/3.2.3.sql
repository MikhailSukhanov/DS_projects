UPDATE attempt
SET result = (SELECT ROUND(SUM(IF(is_correct, 1, 0))*100/3) AS result
              FROM answer
              INNER JOIN testing USING (answer_id)
              WHERE attempt_id = 8)
WHERE attempt_id = 8;
SELECT * FROM attempt
