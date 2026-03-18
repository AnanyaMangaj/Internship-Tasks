#Mini project amazon sales data 

CREATE TABLE customers02 (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    city VARCHAR(50),
    state VARCHAR(50),
    sign_up_date DATE
);

INSERT INTO customers02 VALUES
(1,'Ravi Kumar','Bangalore','Karnataka','2023-01-10'),
(2,'Anita Sharma','Mumbai','Maharashtra','2023-02-15'),
(3,'Rahul Singh','Delhi','Delhi','2023-03-12'),
(4,'Priya Nair','Kochi','Kerala','2023-04-08'),
(5,'Arjun Patel','Ahmedabad','Gujarat','2023-05-20'),
(6,'Neha Gupta','Lucknow','Uttar Pradesh','2023-06-11'),
(7,'Vikram Reddy','Hyderabad','Telangana','2023-07-03'),
(8,'Sneha Das','Kolkata','West Bengal','2023-08-19');

select * from customers02


CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);

INSERT INTO products VALUES
(101,'Laptop','Electronics',65000),
(102,'Smartphone','Electronics',30000),
(103,'Headphones','Electronics',2000),
(104,'Office Chair','Furniture',7000),
(105,'Running Shoes','Sports',4500),
(106,'Backpack','Accessories',1200),
(107,'Smart Watch','Electronics',9000),
(108,'Gaming Mouse','Electronics',2500);

select * from products


CREATE TABLE orders01 (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers02(id)
);

INSERT INTO orders01 VALUES
(1001,1,'2024-01-05',65000),
(1002,2,'2024-01-08',32000),
(1003,3,'2024-01-10',4500),
(1004,4,'2024-01-15',7000),
(1005,5,'2024-01-18',9000),
(1006,6,'2024-02-02',2000),
(1007,7,'2024-02-05',2500),
(1008,8,'2024-02-10',1200),
(1009,1,'2024-02-15',30000),
(1010,2,'2024-02-18',4500);

select * from orders01


CREATE TABLE orderitems01 (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders01(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

INSERT INTO orderitems01 VALUES
(1,1001,101,1,65000),
(2,1002,102,1,30000),
(3,1002,103,1,2000),
(4,1003,105,1,4500),
(5,1004,104,1,7000),
(6,1005,107,1,9000),
(7,1006,103,1,2000),
(8,1007,108,1,2500),
(9,1008,106,1,1200),
(10,1009,102,1,30000),
(11,1010,105,1,4500);

select * from orderitems01

#1.Total revenue
select sum(total_amount) as total_revenue from orders01

#2.Total orders 
select count(order_id) as toal_orders from orders01

#3.Average order value
select avg(total_amount) as average_order_value from orders01

#4.Top 5 selling products
SELECT 
    p.product_name,
    SUM(oi.quantity) AS total_sold
FROM orderitems01 oi
JOIN products p 
ON oi.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_sold DESC
LIMIT 5;

#5.top customers
select c.name , sum(o.total_amount) as total_spent from customers02 c join orders01 o on c.id = o.customer_id group by c.name order by total_spent desc limit 5

#6.Sales by customer
SELECT 
    c.name,
    SUM(o.total_amount) AS sales
FROM customers02 c
JOIN orders01 o
ON c.id = o.customer_id
GROUP BY c.name;

#7.Sales by month
SELECT 
    MONTH(order_date) AS month,
    SUM(total_amount) AS monthly_sales
FROM orders01
GROUP BY MONTH(order_date)
ORDER BY month;

#8.Sales by region
SELECT 
    c.state,
    SUM(o.total_amount) AS regional_sales
FROM customers02 c
JOIN orders01 o
ON c.id = o.customer_id
GROUP BY c.state
ORDER BY regional_sales DESC;