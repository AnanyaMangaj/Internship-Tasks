#task 15- create a teacher class that inherits from the staff
class Staff:
    def __init__(self, name):
        self.name = name
class Teacher(Staff):
    def display(self):
        print("Teacher Name:", self.name)
        
t1 = Teacher("Anita")
t1.display()
