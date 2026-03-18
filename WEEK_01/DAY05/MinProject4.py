#Mini project4 : Employee management system
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary   # private salary
    def calculate_salary(self):
        return self.__salary
class Manager(Employee):
    def calculate_salary(self):
        return super().calculate_salary() + 10000
class Intern(Employee):
    def calculate_salary(self):
        return super().calculate_salary() - 5000

employees = []
employees.append(Manager("Rahul", 50000))
employees.append(Intern("Neha", 30000))
employees.append(Manager("Amit", 55000))
print("Payroll Details:")
for emp in employees:
    print(emp.name, "-", emp.calculate_salary())
