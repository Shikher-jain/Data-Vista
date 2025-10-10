CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    dept VARCHAR(20)
);

CREATE TABLE orders ( 
    order_id INT,
    emp_id INT,
    amount DECIMAL(10,2),
);

CREATE TABLE products (
    product_id INT,
    product_name VARCHAR(50),
    price DECIMAL(10,2)
);


INSERT INTO employees (id, name, dept) VALUES
(1, 'Alice', 'HR'),
(2, 'Bob', 'IT'),
(3, 'Charlie', 'Finance'),
(4, 'David', 'IT'),
(5, 'Eve', 'HR');

INSERT INTO orders (order_id, emp_id, amount, ) VALUES
(101, 1, 250),
(102, 2, 540),
(103, 3, 120),
(104, 4, 310),
(105, 5, 450);

INSERT INTO products (product_id, product_name, price) VALUES
(201, 'Laptop', 1200.00), 
(202, 'Mouse', 25.50),
(203, 'Keyboard', 45.00),
(204, 'Monitor', 300.00),
(205, 'Printer', 150.75);
