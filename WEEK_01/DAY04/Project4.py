#Project 4:user login system
with open("users.txt","w") as f:
    f.write("admin,1234\n")
    f.write("user1,abcd\n")
    f.write("test,9999\n")

print("users.txt created")

def login(username, password):
    try:
        with open("users.txt", "r") as file:
            for line in file:
                user, pwd = line.strip().split(",")
                if user == username and pwd == password:
                    print("Login Successful")
                    return
        print("Invalid username or password")
    except FileNotFoundError:
        print("User file not found")
        
login("admin", "1234")
