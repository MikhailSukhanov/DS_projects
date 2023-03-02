SELECT name_student, date_attempt, result
FROM subject
     INNER JOIN attempt USING (subject_id)
     INNER JOIN student USING (student_id)
WHERE name_subject = "Основы баз данных"
ORDER BY result DESC
