#task3 :Login authentication system(logical+conditional statement)
cred={"alice" : 123, "bob" : 456, "charli": 789}
a = input("Enter username:")
b = int(input("Enter password:"))
if( a in cred and cred[a] == b):
    print("Login succcessful")
else :
    print("Login unsuccessful")
