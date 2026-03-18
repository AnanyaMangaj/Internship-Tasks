#mini project
def calculate_net_salary(basic_pay):

    hra_amount = basic_pay * 0.20
    pf_amount = basic_pay * 0.12

    final_pay = basic_pay + hra_amount - pf_amount
    return final_pay


salary = float(input("Enter basic salary: "))

result = calculate_net_salary(salary)

print("Net Salary:", result)
