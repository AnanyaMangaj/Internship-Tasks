#task6 : user login and role validation system
users = ["admin", "staff", "student"]
roles = ["manager", "employee"]

username = input("Enter username: ")
role = input("Enter role: ")

if username in users:
    if role in roles:
        print("Login successful")
    else:
        print("Invalid role")
else:
    print("User not found")
