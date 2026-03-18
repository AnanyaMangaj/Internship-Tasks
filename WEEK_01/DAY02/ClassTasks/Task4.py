#task4:Shooping discount engine(Assignment operator)
a = float(input("Enter the cart value"))
if( a > 10000):
    b = a - (a * 0.35)
    print("Discounted price is", b )
else :
    print("Discount not applicable")
