#task7 - create a user class where password is private and allow password update using a setter method
class User:
    def __init__(self, username, password):
        self.username = username
        self.__password = password   # private password
    def set_password(self, new_password):
        self.__password = new_password
        print("Password updated successfully")
    def show_user(self):
        print("Username:", self.username)

u1 = User("admin", "1234")
u1.show_user()
u1.set_password("abcd123")
