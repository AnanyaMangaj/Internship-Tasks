#task3: generate daily report automatically
file = open("sales.txt", "w")
file.write("100\n")
file.write("200\n")
file.write("150\n")
file.close()

print("sales.txt created")

def generate_report():
    total_sales = 0
    try:
        file = open("sales.txt", "r")
        for line in file:
            total_sales = total_sales + int(line.strip())
        file.close()
        report = open("report.txt", "w")
        report.write("Daily Sales Report\n")
        report.write("Total Sales: " + str(total_sales))
        report.close()
        print("Report generated successfully")
    except:
        print("Error reading sales file")

generate_report()
