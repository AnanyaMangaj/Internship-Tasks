#task 11 - create a base class vehicle with a method start()
class Vehicle:
    def start(self):
        print("Vehicle started")
class Car(Vehicle):
    pass
class Bus(Vehicle):
    pass
class Bike(Vehicle):
    pass
    
c = Car()
b = Bus()
bk = Bike()
c.start()
b.start()
bk.start()
