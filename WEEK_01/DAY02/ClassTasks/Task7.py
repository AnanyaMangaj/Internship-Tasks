#task7: Even-Odd Transaction Checker


transaction_id = int(input("Enter Transaction ID: "))

if transaction_id % 2 == 0:
    print("Transaction ID", transaction_id, "is EVEN.")
else:
    print("Transaction ID", transaction_id, "is ODD.")
