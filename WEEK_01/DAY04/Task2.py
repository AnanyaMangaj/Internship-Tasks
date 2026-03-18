#task02 : task 02: read the csv data (employees),filter high salary employees,write the result to new file
file = open("employees.csv", "w")
file.write("Ram,45000\n")
file.write("Sita,60000\n")
file.write("John,52000\n")
file.write("Arun,30000\n")
file.close()

print("employees.csv created")


def filter_high_salary(filename):

    try:
        file = open(filename, "r")
        output = open("high_salary_employees.txt", "w")

        for line in file:
            data = line.strip().split(",")

            name = data[0]
            salary = int(data[1])

            if salary >= 50000:
                output.write(name + "," + str(salary) + "\n")

        file.close()
        output.close()

        print("High salary employees saved to high_salary_employees.txt")

    except:
        print("File not found or error reading file")


filter_high_salary("employees.csv")
