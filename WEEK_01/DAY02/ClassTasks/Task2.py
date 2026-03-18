#task2:internship eligibility checker
a = float(input("enter the CGPA"))
print("CGPA is:", a)
b = int(input("enter passing year"))
print("Passing year is:", b)
c = float(input("enter the skill test score"))
print("Skill test score is:", c)
if(a >= 8.0 and b == 2026 and c >= 8.5):
    print("Eligible for internship")
else :
    print("Not eligible for internship")
