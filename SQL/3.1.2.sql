SELECT name_subject, COUNT(result) AS Количество, ROUND(AVG(result), 2) AS Среднее
FROM subject
     LEFT JOIN attempt USING (subject_id)
GROUP BY name_subject
ORDER BY ROUND(AVG(result), 2) DESC
