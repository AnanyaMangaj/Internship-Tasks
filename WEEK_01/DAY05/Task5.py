#task5 - create a class book that stores title and the author name
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    def display(self):
        print(self.title, "-", self.author)

b1 = Book("Python Basics", "John")
b2 = Book("Data Science", "Emma")
b1.display()
b2.display()

