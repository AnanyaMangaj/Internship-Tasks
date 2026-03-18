#Mini project2 : build a student result system with grading logic
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks
    def calculate_grade(self):
        if self.marks >= 80:
            grade = "A"
        elif self.marks >= 60:
            grade = "B"
        elif self.marks >= 40:
            grade = "C"
        else:
            grade = "Fail"

        print("Name:", self.name)
        print("Marks:", self.marks)
        print("Grade:", grade)

s1 = Student("Rahul", 75)
s2 = Student("Neha", 88)
s3 = Student("Amit", 35)
s1.calculate_grade()
print()
s2.calculate_grade()
print()
s3.calculate_grade()
