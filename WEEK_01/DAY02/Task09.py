#task9 : Attendance Status System
present_students = ["ram", "sita", "john"]
student_name = input("Enter student name: ").lower()
if student_name in present_students:
    print("Student is Present")
else:
    print("Student is Absent")
