SELECT name, city, date_first, ((DATEDIFF(date_last, date_first) + 1) * per_diem) AS Сумма
FROM trip
WHERE MONTHNAME(date_first) IN ('February', 'March')
ORDER BY name, ((DATEDIFF(date_last, date_first) + 1) * per_diem) DESC
