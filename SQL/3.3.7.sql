SELECT name_department, name_program, plan, COUNT(enrollee_id) AS Количество, ROUND(COUNT(enrollee_id)/plan, 2) AS Конкурс
FROM department
     INNER JOIN program USING (department_id)
     INNER JOIN program_enrollee USING (program_id)
GROUP BY name_department, name_program, plan
ORDER BY Конкурс DESC
