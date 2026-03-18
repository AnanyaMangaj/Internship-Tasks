#task 3 - create a rectangle class with length and width
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
    def area(self):
        return self.length * self.width
    def perimeter(self):
        return 2 * (self.length + self.width)

r1 = Rectangle(10, 5)
print("Area:", r1.area())
print("Perimeter:", r1.perimeter())
