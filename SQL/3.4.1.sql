CREATE TABLE applicant AS
SELECT program_enrollee.program_id, program_enrollee.enrollee_id, SUM(result) AS itog
FROM program_enrollee
     INNER JOIN program_subject ON program_subject.program_id = program_enrollee.program_id
     INNER JOIN enrollee_subject ON enrollee_subject.subject_id = program_subject.subject_id AND enrollee_subject.enrollee_id = program_enrollee.enrollee_id
GROUP BY program_id, enrollee_id
ORDER BY program_id, itog DESC;
SELECT * FROM applicant
