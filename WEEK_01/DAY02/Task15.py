#task15: User Access Review System
users = {
    "ram": "admin",
    "sita": "user",
    "john": "admin",
    "arun": "user"
}

print("Users with Admin Access:")

for name in users:
    if users[name] == "admin":
        print(name)
