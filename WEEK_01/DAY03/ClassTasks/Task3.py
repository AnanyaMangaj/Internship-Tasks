#task 03:square&cube calculator(lambda+map)
nums = [2, 3, 4, 5]
square = list(map(lambda x: x * x, nums))
cube = list(map(lambda x: x * x * x, nums))
print("Squares:", square)
print("Cubes:", cube)
