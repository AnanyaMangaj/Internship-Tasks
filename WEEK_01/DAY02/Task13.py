#task13 : Product Inventory Checker
products = {
    "laptop": "available",
    "mobile": "out of stock",
    "keyboard": "available",
    "mouse": "out of stock"
}
print("Available Products:")
for item in products:
    if products[item] == "available":
        print(item)
