SELECT name_city, COUNT(buy_id) AS Количество
FROM city
     INNER JOIN client ON city.city_id = client.city_id
     INNER JOIN buy ON client.client_id = buy.client_id
GROUP BY name_city
ORDER BY Количество DESC, name_city
