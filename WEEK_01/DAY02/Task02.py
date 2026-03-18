#task2: Inrernship eligibility checker
# internship task

cgpa = float(input("Enter CGPA: "))
year = int(input("Enter year of study: "))
skill = int(input("Enter skill test score: "))

if cgpa >= 7 and year == 4 and skill >= 60:
    print("Eligible for internship")
else:
    print("Not eligible for internship")
