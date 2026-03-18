#Task 02: ATM withdrawal function
def withdraw(balance, amount):

    if amount <= balance and amount % 100 == 0:
        balance = balance - amount
        return balance
    else:
        return "Withdrawal not allowed"

x = withdraw(10000, 100)
print(x)
