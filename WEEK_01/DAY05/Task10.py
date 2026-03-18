#task 10 - create a laptop class that restricts direct access to the serial number
class Laptop:
    def __init__(self, brand, serial_no):
        self.brand = brand
        self.__serial_no = serial_no   # private serial number
    def get_serial(self):
        return self.__serial_no
l1 = Laptop("Dell", "DL12345")
print("Brand:", l1.brand)
print("Serial Number:", l1.get_serial())
