SELECT name_program, plan
FROM program
WHERE plan = (SELECT plan FROM program ORDER BY plan DESC LIMIT 1)
