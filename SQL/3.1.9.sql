SELECT name_subject, CONCAT(SUBSTRING(name_question, 1, 30), "...") AS Вопрос, COUNT(is_correct) AS Всего_ответов, ROUND((SUM(is_correct)/COUNT(is_correct))*100, 2) AS Успешность
FROM subject
     INNER JOIN attempt ON subject.subject_id = attempt.subject_id
     INNER JOIN testing ON attempt.attempt_id = testing.attempt_id
     INNER JOIN answer ON answer.answer_id = testing.answer_id
     INNER JOIN question ON question.question_id = answer.question_id
GROUP BY name_subject, Вопрос
ORDER BY name_subject, Успешность DESC, Вопрос
