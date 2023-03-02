SELECT name_student, name_subject, date_attempt, ROUND((SUM(is_correct)*100/3), 2) AS Результат
FROM student
     INNER JOIN attempt ON student.student_id = attempt.student_id 
     INNER JOIN subject ON subject.subject_id = attempt.subject_id
     INNER JOIN testing ON testing.attempt_id = attempt.attempt_id
     INNER JOIN answer ON answer.answer_id = testing.answer_id
GROUP BY name_student, name_subject, date_attempt
ORDER BY name_student, date_attempt DESC
