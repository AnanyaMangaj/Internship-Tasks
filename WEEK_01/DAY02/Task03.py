#task03: E-commerce discount logic
cart_value = float(input("Enter cart value: "))

if cart_value >= 5000:
    discount = cart_value * 0.20
    final_amount = cart_value - discount
    print("20% discount applied")
    print("Final amount:", final_amount)

elif cart_value >= 3000:
    discount = cart_value * 0.10
    final_amount = cart_value - discount
    print("10% discount applied")
    print("Final amount:", final_amount)

else:
    print("No discount available")
    print("Final amount:", cart_value)
