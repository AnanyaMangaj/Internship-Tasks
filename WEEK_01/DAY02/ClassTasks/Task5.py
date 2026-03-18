#task5:course enrollment validator(membership operator+conditional statement)

student_name = input("Enter student name: ")
course = input("Enter enrolled course: ").lower()

available_courses = ["python", "java", "data science", "web development"]

if course in available_courses:
    if course == "python":
        print(student_name, "is enrolled in Python course.")
    else:
        print(student_name, "is enrolled in", course, "course, not Python.")
else:
    print("Entered course is not available.")

    
