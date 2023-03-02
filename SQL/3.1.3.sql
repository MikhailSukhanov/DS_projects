SELECT name_student, result
FROM student
     INNER JOIN attempt USING (student_id)
WHERE result = (SELECT result FROM attempt ORDER BY result DESC LIMIT 1)
GROUP BY name_student, result
