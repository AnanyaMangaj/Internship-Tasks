#task 19 - create a function that accepts object and calls display method
class Student:
    def display(self):
        print("Student details displayed")
class Teacher:
    def display(self):
        print("Teacher details displayed")
def show_details(obj):
    obj.display()
s = Student()
t = Teacher()
show_details(s)
show_details(t)
