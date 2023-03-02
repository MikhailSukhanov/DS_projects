SET @num_pr := 0;
SET @row_num := 1;

UPDATE applicant_order
       INNER JOIN (SELECT *, IF(program_id = @num_pr, @row_num := @row_num + 1, @row_num := 1) AS str_num, @num_pr := program_id AS add_var FROM applicant_order) AS first_01 ON applicant_order.program_id = first_01.program_id AND applicant_order.enrollee_id = first_01.enrollee_id
SET applicant_order.str_id = str_num;
SELECT * FROM applicant_order
