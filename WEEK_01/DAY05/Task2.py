#task2 - write a class with class name mobile with attributes brand,price
class Mobile:
    def __init__(self, brand, price):
        self.brand = brand
        self.price = price
    def display(self):
        print(self.brand, self.price)
m1 = Mobile("Samsung", 25000)
m2 = Mobile("Apple", 80000)
m1.display()
m2.display()
