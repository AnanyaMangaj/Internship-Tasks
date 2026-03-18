#task1:Employee salary calculator(Arithmetic operator)
salary = float(input("Enter basic salary: "))
extra_bonus = float(input("Enter bonus: "))
deduction = float(input("Enter tax: "))

total = salary + extra_bonus
net_amount = total - deduction
print("Your Monthly Salary is:", net_amount)
