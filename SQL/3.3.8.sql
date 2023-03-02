SELECT name_program
FROM program
     INNER JOIN program_subject USING (program_id)
     INNER JOIN subject USING (subject_id)
WHERE name_subject IN ("Информатика", "Математика")
GROUP BY name_program
HAVING COUNT(name_subject) = 2
ORDER BY name_program
