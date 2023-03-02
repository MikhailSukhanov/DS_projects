INSERT INTO testing (attempt_id, question_id)
SELECT attempt_id, question_id
FROM question
     INNER JOIN attempt USING (subject_id)
WHERE subject_id = (SELECT subject_id FROM attempt ORDER BY attempt_id DESC LIMIT 1) AND
      attempt_id = (SELECT attempt_id FROM attempt ORDER BY attempt_id DESC LIMIT 1)
ORDER BY RAND()
LIMIT 3;
SELECT * FROM testing
