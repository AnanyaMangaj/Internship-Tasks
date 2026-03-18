#task01: count the number of errors
file = open("log.txt", "w")

file.write("System started\n")
file.write("Error: connection failed\n")
file.write("Login successful\n")
file.write("Error: timeout occurred\n")

file.close()

print("File created")


def count_errors(filename):
    error_lines = []
    count = 0
    try:
        file = open(filename, "r")
        for line in file:
            if "error" in line.lower():
                count = count + 1
                error_lines.append(line)
        file.close()
        log = open("error_log.txt", "w")
        for e in error_lines:
            log.write(e)
        log.close()
        return count
    except:
        return "File not found"

print("Total Errors:", count_errors("log.txt"))
