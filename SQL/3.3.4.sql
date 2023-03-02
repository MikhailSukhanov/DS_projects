SELECT name_program
FROM program
     INNER JOIN program_subject USING (program_id)
GROUP BY name_program
HAVING MIN(min_result) >= 40
ORDER BY name_program
