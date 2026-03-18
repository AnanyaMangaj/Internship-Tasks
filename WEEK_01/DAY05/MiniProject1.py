#Mini project1:build a library management system using class, inheritance, encapsulation
class LibraryItem:
    def __init__(self, title):
        self.title = title
class Book(LibraryItem):
    def __init__(self, title, author):
        LibraryItem.__init__(self, title)
        self.author = author
        self.__available = True   # private variable
    def borrow(self):
        if self.__available:
            self.__available = False
            print("Book borrowed:", self.title)
        else:
            print("Book already borrowed")
    def return_book(self):
        self.__available = True
        print("Book returned:", self.title)
    def show_status(self):
        if self.__available:
            print(self.title, "is available")
        else:
            print(self.title, "is not available")

b1 = Book("Python Basics", "John")
b1.show_status()
b1.borrow()
b1.show_status()
b1.return_book()
b1.show_status()
