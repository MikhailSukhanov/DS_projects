UPDATE fine AS f, payment AS p
SET f.date_payment = p.date_payment, sum_fine = IF(DATEDIFF(p.date_payment, p.date_violation) <= 20, sum_fine * 0.5, sum_fine)
WHERE f.name = p.name AND f.number_plate=p.number_plate AND f.violation = p.violation AND f.date_violation = p.date_violation;
SELECT name, violation, sum_fine, date_violation, date_payment FROM fine
