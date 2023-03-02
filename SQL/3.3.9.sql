SELECT name_program, name_enrollee, SUM(result) AS itog
FROM enrollee_subject
     INNER JOIN enrollee ON enrollee.enrollee_id = enrollee_subject.enrollee_id
     INNER JOIN program_enrollee ON program_enrollee.enrollee_id = enrollee.enrollee_id
     INNER JOIN program ON program.program_id = program_enrollee.program_id
     INNER JOIN program_subject ON program_subject.program_id = program.program_id AND program_subject.subject_id = enrollee_subject.subject_id
GROUP BY name_program, name_enrollee
ORDER BY name_program, itog DESC
