UPDATE applicant
       INNER JOIN (SELECT enrollee_id, IF(SUM(bonus) IS NULL, 0, SUM(bonus)) AS bonus
                   FROM achievement
                   RIGHT JOIN enrollee_achievement USING (achievement_id)
                   GROUP BY enrollee_id) AS First_01 USING (enrollee_id)
SET itog = itog + bonus;
SELECT * FROM applicant
