#Project1 : student record management
def add_student(name):
    try:
        file = open("students.txt", "a")
        file.write(name + "\n")
        file.close()
        print("Student added")
    except:
        print("Error adding student")

def view_students():
    try:
        file = open("students.txt", "r")
        print(file.read())
        file.close()
    except:
        print("File not found")

def search_student(name):
    found = False
    try:
        file = open("students.txt", "r")
        for line in file:
            if name.lower() in line.lower():
                print("Student Found:", line.strip())
                found = True
        file.close()
        if found == False:
            print("Student not found")
    except:
        print("File not found")

add_student("Ayaan")
add_student("Rahul")
view_students()
search_student("Ayaan")
