#task1 - create a student class with attributes name,rollno,marks
class StudentInfo:

    def __init__(self, student_name, roll_number, score):
        self.student_name = student_name
        self.roll_number = roll_number
        self.score = score

    def show_details(self):
        print("Name:", self.student_name)
        print("Roll No:", self.roll_number)
        print("Marks:", self.score)
        print()


st1 = StudentInfo("Rahul", 11, 82)
st2 = StudentInfo("Neha", 12, 88)
st3 = StudentInfo("Karan", 13, 75)

st1.show_details()
st2.show_details()
st3.show_details()
