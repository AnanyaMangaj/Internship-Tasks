#task 13 - create account class--inherit into Saving account and add intrest calculation
class Account:
    def __init__(self, balance):
        self.balance = balance
class SavingAccount(Account):
    def calculate_interest(self):
        interest = self.balance * 0.05
        print("Interest:", interest)

sa = SavingAccount(10000)
sa.calculate_interest()
