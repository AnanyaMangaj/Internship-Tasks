#task 12 - create person class with name and age inherit into employee and add employee_id
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
class Employee(Person):
    def __init__(self, name, age, employee_id):
        Person.__init__(self, name, age)
        self.employee_id = employee_id
    def display(self):
        print("Name:", self.name)
        print("Age:", self.age)
        print("Employee ID:", self.employee_id)

e1 = Employee("Rahul", 25, 101)
e1.display()
