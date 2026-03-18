#task 20 - create a payment system where pay() works: credit card,emi,upi,cash,debit card
class CreditCard:
    def pay(self):
        print("Payment done using Credit Card")
class EMI:
    def pay(self):
        print("Payment done using EMI")
class UPI:
    def pay(self):
        print("Payment done using UPI")
class Cash:
    def pay(self):
        print("Payment done using Cash")
class DebitCard:
    def pay(self):
        print("Payment done using Debit Card")

c = CreditCard()
e = EMI()
u = UPI()
ca = Cash()
d = DebitCard()
c.pay()
e.pay()
u.pay()
ca.pay()
d.pay()
