#task 04:discount calculator
price = float(input("Enter product price: "))
discount = lambda p: p - (p * 0.10)
final_price = discount(price)
print("Final Price after discount:", final_price)
