SELECT name_program, name_enrollee
FROM enrollee
     INNER JOIN program_enrollee ON program_enrollee.enrollee_id = enrollee.enrollee_id
     INNER JOIN program ON program.program_id = program_enrollee.program_id
     INNER JOIN program_subject ON program_subject.program_id = program.program_id
     INNER JOIN enrollee_subject ON enrollee_subject.subject_id = program_subject.subject_id AND enrollee_subject.enrollee_id = enrollee.enrollee_id
WHERE result/min_result < 1
GROUP BY name_program, name_enrollee
ORDER BY name_program, name_enrollee
