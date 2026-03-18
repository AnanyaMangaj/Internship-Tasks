#task4 - create a class employee with name and salary and exceute their objects and print the output
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    def display(self):
        print(self.name, self.salary)

e1 = Employee("Rahul", 35000)
e2 = Employee("Neha", 42000)
e1.display()
e2.display()
