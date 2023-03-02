CREATE TABLE student AS
SELECT name_program, name_enrollee, itog 
FROM enrollee
     INNER JOIN applicant_order USING (enrollee_id)
     INNER JOIN program USING (program_id)
WHERE str_id <= plan
ORDER BY name_program, itog DESC;
SELECT * FROM student
