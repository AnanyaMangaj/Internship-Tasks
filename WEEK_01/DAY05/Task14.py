#task 14 - create a shape class and inherit circle and rectangle
class Shape:
    def show(self):
        print("This is a shape")
class Circle(Shape):
    def area(self, radius):
        print("Circle Area:", 3.14 * radius * radius)
class Rectangle(Shape):
    def area(self, length, width):
        print("Rectangle Area:", length * width)

c = Circle()
r = Rectangle()
c.show()
c.area(5)
r.show()
r.area(10, 4)
