#Project 3 : Atm simulation
with open("employees.csv","w") as f:
    f.write("name,salary\n")
    f.write("Ram,45000\n")
    f.write("Sita,60000\n")
    f.write("John,52000\n")
    f.write("Arun,30000\n")

print("employees.csv created")

import csv
def salary_analyzer(filename):
    try:
        salaries = []
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                salaries.append(int(row["salary"]))
        highest = max(salaries)
        lowest = min(salaries)
        average = sum(salaries) / len(salaries)
        print("Highest Salary:", highest)
        print("Lowest Salary:", lowest)
        print("Average Salary:", average)
    except FileNotFoundError:
        print("Employees file not found")
    except ValueError:
        print("Invalid salary data")


salary_analyzer("employees.csv")
