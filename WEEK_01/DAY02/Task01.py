#task01 : user acess login control
users = {
    "alice": "active",
    "bob": "inactive",
    "charlie": "active"
}
username = input("Enter username: ")
if username in users:
    status = users[username]

    if status == "active":
        print("Login successful. Welcome,", username)
    else:
        print("Account is inactive. Access denied.")

else:
    print("User does not exist.")
