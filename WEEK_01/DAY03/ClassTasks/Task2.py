#task 02:sort employees by salary(lambda)
employees = [
    ("Ram", 30000),
    ("Sita", 45000),
    ("John", 35000)
]
sorted_employees = sorted(employees, key=lambda x: x[1])
print("Sorted Employees by Salary:")
print(sorted_employees)
