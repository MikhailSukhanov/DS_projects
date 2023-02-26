INSERT INTO author (name_author)
SELECT supply.author
FROM author 
     RIGHT JOIN supply ON author.name_author = supply.author
WHERE author.name_author IS NULL;
SELECT * FROM author
