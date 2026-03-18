#task 9 - create a class atm that hides the pin and allow the transaction only after validation
class ATM:
    def __init__(self, pin, balance):
        self.__pin = pin             # private pin
        self.balance = balance
    def validate_pin(self, entered_pin):
        if entered_pin == self.__pin:
            return True
        else:
            return False
    def withdraw(self, entered_pin, amount):
        if self.validate_pin(entered_pin):
            if amount <= self.balance:
                self.balance = self.balance - amount
                print("Withdrawal successful")
                print("Remaining balance:", self.balance)
            else:
                print("Insufficient balance")
        else:
            print("Invalid PIN. Transaction denied.")

atm = ATM(1234, 5000)
atm.withdraw(1234, 2000)
