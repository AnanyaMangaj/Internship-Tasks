#task6:Role verification system(identity operator+conditional statement)
a = input("Enter username")
b = input("Enter role")
usernames =["alice", "bob", "jack"]
roles = ["manager", "hr", "developer"]
if( a in usernames and b in roles ):
    print("Login successful")
else :
    print("Login not successful")
