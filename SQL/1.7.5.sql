UPDATE fine AS f, (SELECT name, number_plate, violation FROM fine GROUP BY name, number_plate, violation HAVING COUNT(*) >= 2) AS query_in
SET f.sum_fine = f.sum_fine * 2
WHERE f.name = query_in.name AND f.number_plate = query_in.number_plate AND f.violation = query_in.violation AND date_payment IS NULL;
SELECT * FROM fine
