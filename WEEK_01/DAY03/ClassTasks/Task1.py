#task 01:even number filter(lambda)
numbers = [10, 15, 20, 25, 30, 33]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print("Even Numbers:", even_numbers)
