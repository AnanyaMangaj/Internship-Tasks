#task 18 - create multiple classes with method login() behaving different
class Admin:
    def login(self):
        print("Admin logged in with full access")
class User:
    def login(self):
        print("User logged in with limited access")
class Guest:
    def login(self):
        print("Guest logged in with view only access")

a = Admin()
u = User()
g = Guest()
a.login()
u.login()
g.login()
