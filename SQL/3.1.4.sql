SELECT name_student, name_subject, DATEDIFF(MAX(date_attempt),MIN(date_attempt)) AS Интервал
FROM subject
     INNER JOIN attempt USING (subject_id)
     INNER JOIN student USING (student_id)
GROUP BY name_student, name_subject
HAVING COUNT(result) > 1
ORDER BY Интервал
