#Task 01:student grade calculator
def calculate_grade(marks):
    total = 0
    for m in marks:
        total = total + m

    average = total / len(marks)
    if average >= 80:
        return "A"
    elif average >= 60:
        return "B"
    elif average >= 40:
        return "C"
    else:
        return "Fail"

student_marks = [88, 68, 80, 90]
grade = calculate_grade(student_marks)
print("Grade:", grade)
