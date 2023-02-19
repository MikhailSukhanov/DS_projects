UPDATE fine AS f, traffic_violation AS tv
SET f.sum_fine = tv.sum_fine
WHERE (f.sum_fine is NULL) AND (f.violation = tv.violation);
SELECT * FROM fine
