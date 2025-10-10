CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    dept VARCHAR(20),
    email VARCHAR(100) -- new column added
);

CREATE TABLE orders (
    order_id INT,
    emp_id INT,
    amount FLOAT,  -- datatype changed from DECIMAL to FLOAT
    status VARCHAR(20) -- new column added
);

CREATE TABLE products (
    product_id INT,
    product_name VARCHAR(50),
    price DECIMAL(10,2)
);

CREATE TABLE departments (   -- new table added
    dept_id INT,
    dept_name VARCHAR(50)
);

-- Employees
INSERT INTO employees (id, name, dept, email) VALUES
(1, 'Alice', 'HR', 'alice@example.com'),
(2, 'Bob', 'IT', 'bob@example.com'),
(3, 'Charlie', 'Finance', 'charlie@example.com'),
(4, 'David', 'IT', 'david@example.com'),
(5, 'Eve', 'HR', 'eve@example.com');

-- Orders
INSERT INTO orders (order_id, emp_id, amount, status) VALUES
(101, 1, 250.75, 'Pending'),
(102, 2, 540.50, 'Completed'),
(103, 3, 120.00, 'Cancelled'),
(104, 4, 310.40, 'Pending'),
(105, 5, 450.00, 'Completed');

-- Products
INSERT INTO products (product_id, product_name, price) VALUES
(201, 'Laptop', 1200.00),
(202, 'Mouse', 25.50),
(203, 'Keyboard', 45.00),
(204, 'Monitor', 300.00),
(205, 'Printer', 150.75);

-- Departments
INSERT INTO departments (dept_id, dept_name) VALUES
(1, 'HR'),
(2, 'IT'),
(3, 'Finance'),
(4, 'Marketing'),
(5, 'Sales');
