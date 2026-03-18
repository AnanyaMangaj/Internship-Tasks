#task11 :Internship Shortlisting System
students = [
    {"name": "ram", "cgpa": 7.5},
    {"name": "sita", "cgpa": 6.8},
    {"name": "john", "cgpa": 7.2},
    {"name": "arun", "cgpa": 6.5}
]
print("Shortlisted Students:")
for student in students:
    if student["cgpa"] >= 7:
        print(student["name"])
