#task8 :  Bank Account Balance Validation
account = {
    "balance": float(input("Enter account balance: ")),
    "withdraw": float(input("Enter withdrawal amount: "))
}

if account["withdraw"] <= account["balance"]:
    print("Withdrawal successful")
    remaining = account["balance"] - account["withdraw"]
    print("Remaining balance:", remaining)
else:
    print("Insufficient balance")
