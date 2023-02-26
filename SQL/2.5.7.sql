INSERT INTO buy_step (buy_id, step_id)
SELECT (SELECT buy_id FROM buy ORDER BY buy_id DESC LIMIT 1), step_id
FROM step;
SELECT * FROM buy_step
