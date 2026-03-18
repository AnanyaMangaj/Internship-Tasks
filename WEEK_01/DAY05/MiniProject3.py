#Mini project3 : build a simple e-commerce product system
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    def show_product(self):
        print("Product Name:", self.name)
        print("Product Price:", self.price)

p1 = Product("Laptop", 55000)
p2 = Product("Mobile", 20000)
p1.show_product()
print()
p2.show_product()
