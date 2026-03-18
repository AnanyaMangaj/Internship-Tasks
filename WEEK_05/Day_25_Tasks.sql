#Day 25 Tasks
#Task1
create table product_table01(product_id int primary key,product_name varchar(50) not null, price int check(price > 0), category varchar(20) default 'cloths')

insert into product_table01(product_id, product_name,price) values(101,'bat', 15)
insert into product_table01 values(102,'ball', 20,'balling')

select * from product_table01

#Task2
CREATE TABLE departments01 (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employee01 (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(50),
    emp_age INT CHECK (emp_age >= 18),
    email VARCHAR(100) UNIQUE,
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments01(dept_id)
);

INSERT INTO employee01 VALUES
(1,'Rahul',25,'rahul@gmail.com',101),
(2,'Priya',28,'priya@gmail.com',102),
(3,'Amit',30,'amit@gmail.com',101),
(4,'Sneha',27,'sneha@gmail.com',103),
(5,'Arjun',26,'arjun@gmail.com',102);

INSERT INTO employee01 VALUES(6,'Kiran',29,'kiran@gmail.com',101),
(7,'Ravi',31,'ravi@gmail.com',103),
(8,'Pooja',24,'pooja@gmail.com',NULL);

INSERT INTO departments01 VALUES
(101,'IT'),
(102,'HR'),
(103,'FINANCE');

INSERT INTO departments01 VALUES(104,'SALES')


select * from departments01

select * from employee01

INSERT INTO employee01 VALUES
(1,'Ravi',29,'ravi@gmail.com',101);

INSERT INTO employee01 VALUES
(6,'Kiran',24,'rahul@gmail.com',102);

INSERT INTO employee01 VALUES
(7,'Rohit',15,'rohit@gmail.com',101);

INSERT INTO employee01 VALUES
(8,'Sahil',25,'sahil@gmail.com',999);


#Task questions
#1.find the employee name with department name
SELECT e.emp_name, d.dept_name
FROM employee01 e
JOIN departments01 d
ON e.dept_id = d.dept_id;

#2.Find Employees Who Don't Belong to Any Department
SELECT e.emp_name
FROM employee01 e
LEFT JOIN departments01 d
ON e.dept_id = d.dept_id
WHERE d.dept_id IS NULL;

#3.Find Departments Without Employees
SELECT d.dept_name
FROM departments01 d
LEFT JOIN employee01 e
ON d.dept_id = e.dept_id
WHERE e.emp_id IS NULL;

#4.Employee Count per Department
SELECT d.dept_name, COUNT(e.emp_id) AS employee_count
FROM departments01 d
LEFT JOIN employee01 e
ON d.dept_id = e.dept_id
GROUP BY d.dept_name;

#5.find the highest salary in  each department
SELECT d.dept_name, MAX(e.salary) AS highest_salary
FROM employee01 e
JOIN departments01 d
ON e.dept_id = d.dept_id
GROUP BY d.dept_name;

#6.Departments with More Than 2 Employees
SELECT dept_id, COUNT(emp_id) AS total_employees
FROM employee01
GROUP BY dept_id
HAVING COUNT(emp_id) > 2;

#7.Employees Working in Finance Department
SELECT e.emp_name
FROM employee01 e
JOIN departments01 d
ON e.dept_id = d.dept_id
WHERE d.dept_name = 'FINANCE';

#8.Employees Working in IT Department
SELECT e.emp_name
FROM employee01 e
JOIN departments01 d
ON e.dept_id = d.dept_id
WHERE d.dept_name = 'IT';

#9.Employees Working in HR Department
SELECT e.emp_name
FROM employee01 e
JOIN departments01 d
ON e.dept_id = d.dept_id
WHERE d.dept_name = 'HR';