#task01_day24

create table employee(emp_id int primary key,name varchar(100), age int, salary decimal(10,2), dept_id int, hire_date date, manager_id int)

insert into employee (emp_id,name,age, salary, dept_id, hire_date, manager_id) values 
(101, 'Rahul', 28, 50000, 10, '2022-05-12', 201),
(102, 'Anita', 32, 60000, 20, '2021-03-15', 202),
(103, 'Vikram', 29, 55000, 10, '2023-01-20', 201),
(104, 'Priya', 26, 48000, 30, '2022-11-05', 203),
(105, 'Arjun', 35, 75000, 20, '2020-07-18', 202),
(106, 'Sneha', 27, 52000, 30, '2023-02-10', 203),
(107, 'Karan', 31, 62000, 10, '2021-09-25', 201),
(108, 'Meera', 24, 45000, 40, '2023-06-12', 204),
(109, 'Rohit', 30, 58000, 20, '2022-04-22', 202),
(110, 'Pooja', 28, 51000, 30, '2021-12-30', 203),
(111, 'Amit', 33, 67000, 40, '2020-10-10', 204),
(112, 'Neha', 25, 47000, 10, '2023-03-14', 201),
(113, 'Suresh', 36, 80000, 20, '2019-08-01', 202),
(114, 'Divya', 27, 53000, 30, '2022-02-17', 203),
(115, 'Manoj', 34, 72000, 40, '2021-05-09', 204);

select * from employee

#1.retrieve all the employees from the employee table
select * from employee

#2.display employee name and salary only
select name , salary from employee

#3.find the employees whose salary is greater than 50000
select * from employee where salary > 50000 

#4.list employees who is working in the company more than 5 years
select * from employee where hire_date < '2021-01-01'

#5.retrieve the employee who is working in department 201
select * from employee where dept_id = 20

#6.display the employees whose salary is less than 30000
select * from employee where salary < 50000

#7.display employees sorted by salary in descending order
select * from employee order by salary desc

#8.find the average salary of all employees
select avg(salary) from employee 

#9.find the maximum salary in the company
select max(salary) from employee

#10.find the departments having more than 3 employees
select dept_id, count(*) from employee group by dept_id having count(*) > 3

#11.find the employees whose salary is equal to maximum salary
select * from employee where salary = (select max(salary) from employee) 

#12.find the second highest salary
select max(salary) as sec_high_sal from employee where salary <(select max(salary) from employee)

#table01
CREATE TABLE departmentss (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50),
    dept_count INT,
    location VARCHAR(50)
);
INSERT INTO departmentss (dept_id, dept_name, dept_count, location) VALUES
(1, 'HR', 5, 'Bangalore'),
(2, 'Finance', 4, 'Mumbai'),
(3, 'IT', 10, 'Hyderabad'),
(4, 'Sales', 6, 'Delhi'),
(5, 'Marketing', 3, 'Chennai');
select * from departmentss
#table02
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    city VARCHAR(50)
);
INSERT INTO customers (customer_id, customer_name, city) VALUES
(101, 'Rahul', 'Bangalore'),
(102, 'Anita', 'Mumbai'),
(103, 'Vikram', 'Hyderabad'),
(104, 'Priya', 'Delhi'),
(105, 'Arjun', 'Chennai');
select * from customers

#table03
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
INSERT INTO orders (order_id, customer_id, order_date, amount) VALUES
(1, 101, '2024-01-10', 2500),
(2, 102, '2024-02-15', 4000),
(3, 103, '2024-03-05', 3200),
(4, 104, '2024-03-20', 1500),
(5, 105, '2024-04-01', 5000);
select * from orders