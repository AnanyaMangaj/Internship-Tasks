#task7 : Internship Eligibility Checker
student_info = {
    "cgpa": float(input("Enter CGPA: ")),
    "year": int(input("Enter year of study: "))
}

if student_info["cgpa"] >= 7 and student_info["year"] >= 3:
    print("Student is eligible for internship")
else:
    print("Student is not eligible for internship")
