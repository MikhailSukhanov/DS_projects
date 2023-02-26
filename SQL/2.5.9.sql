UPDATE buy_step
SET date_step_end = "2020.04.13"
WHERE buy_id = 5 AND step_id = 1;
UPDATE buy_step
SET date_step_beg = "2020.04.13"
WHERE buy_id = 5 AND step_id = 2;
SELECT * FROM buy_step
