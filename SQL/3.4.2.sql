DELETE FROM applicant
WHERE program_id IN (SELECT program_enrollee.program_id
                     FROM program_enrollee
                          INNER JOIN program_subject ON program_subject.program_id = program_enrollee.program_id
                          INNER JOIN enrollee_subject ON enrollee_subject.subject_id = program_subject.subject_id AND enrollee_subject.enrollee_id = program_enrollee.enrollee_id
                     WHERE result/min_result < 1
                     GROUP BY program_id) AND
      enrollee_id IN (SELECT program_enrollee.enrollee_id
                     FROM program_enrollee
                          INNER JOIN program_subject ON program_subject.program_id = program_enrollee.program_id
                          INNER JOIN enrollee_subject ON enrollee_subject.subject_id = program_subject.subject_id AND enrollee_subject.enrollee_id = program_enrollee.enrollee_id
                     WHERE result/min_result < 1
                     GROUP BY enrollee_id);
SELECT * FROM applicant
