#Task 03 : Password validator
def password_validator(password):
    digit = False
    upper = False
    for ch in password:
        if ch.isdigit():
            digit = True
        if ch.isupper():
            upper = True

    if len(password) >= 8 and digit and upper:
        return "Valid Password"
    else:
        return "Invalid Password"

print(password_validator("Python123"))
print(password_validator("python"))
