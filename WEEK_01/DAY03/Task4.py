#Task 04: Employee salary calculator
def calculate_salary(basic_salary):
    hra = basic_salary * 0.20
    pf = basic_salary * 0.12
    net_salary = basic_salary + hra - pf
    return net_salary

salary = float(input("Enter basic salary: "))
final_salary = calculate_salary(salary)
print("Net Salary:", final_salary)
