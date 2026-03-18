#task8 - create a product class with price as protected variable
class Product:
    def __init__(self, name, price):
        self.name = name
        self._price = price   # protected variable
class DiscountedProduct(Product):
    def show_details(self):
        print("Product Name:", self.name)
        print("Product Price:", self._price)

p1 = DiscountedProduct("Laptop", 55000)
p1.show_details()
