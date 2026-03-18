#task 16 - create a method calculate_salary() in the employee
class Employee:
    def calculate_salary(self):
        print("Calculating employee salary")
class Manager(Employee):
    def calculate_salary(self):
        print("Manager Salary: 60000")
class Intern(Employee):
    def calculate_salary(self):
        print("Intern Salary: 15000")
        
e = Employee()
m = Manager()
i = Intern()
e.calculate_salary()
m.calculate_salary()
i.calculate_salary()
