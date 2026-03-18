#task6 - create a bankaccount class with private balance
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance   # private variable
    def deposit(self, amount):
        self.__balance = self.__balance + amount
        print("Amount deposited")
    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance = self.__balance - amount
            print("Amount withdrawn")
        else:
            print("Insufficient balance")
    def get_balance(self):
        return self.__balance

acc = BankAccount(5000)
acc.deposit(2000)
acc.withdraw(3000)
print("Current Balance:", acc.get_balance())
