SELECT name_question, name_answer, IF(is_correct, "Верно", "Неверно") AS Результат
FROM answer
    INNER JOIN testing USING (answer_id)
    INNER JOIN question ON question.question_id = testing.question_id
WHERE attempt_id = 7
