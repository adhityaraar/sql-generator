SELECT * FROM rucika.RTL;

SELECT SUM(itemQty * itemPrice) AS TotalPurchaseAmount
FROM rucika.RTL
WHERE itemProduct IN ('Item B', ' Item C');

SELECT SUM(itemQty * itemPrice) AS TotalPrice
FROM (
    SELECT itemQty, itemPrice
    FROM rucika.RTL
    WHERE itemProduct IN ('Item A', 'Item G')
    UNION ALL
    SELECT itemQty, itemPrice
    FROM rucika.SJR
    WHERE itemProduct IN ('Item A', 'Item G')
) AS combinedTables;


SELECT COUNT(DISTINCT customer) AS NumberOfCustomers
FROM (
    SELECT customer
    FROM rucika.RTL
    WHERE orderDate >= '2021-02-01' AND orderDate <= '2021-02-28'
    UNION
    SELECT customer
    FROM rucika.SJR
    WHERE orderDate >= '2021-02-01' AND orderDate <= '2021-02-28'
) AS combinedOrders;

SELECT SUM(itemQty) AS TotalQuantity FROM (SELECT itemQty FROM rucika.RTL WHERE itemProduct = 'Item A' UNION ALL SELECT itemQty FROM rucika.SJR WHERE itemProduct = 'Item A') AS combinedTables;

SELECT COUNT(DISTINCT itemProduct) AS TotalItems FROM (SELECT itemProduct FROM rucika.RTL UNION SELECT itemProduct FROM rucika.SJR) AS combinedItems;






