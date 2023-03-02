SELECT name_enrollee
FROM enrollee
     INNER JOIN program_enrollee USING (enrollee_id)
     INNER JOIN program USING (program_id)
WHERE name_program = "Мехатроника и робототехника"
ORDER BY name_enrollee
