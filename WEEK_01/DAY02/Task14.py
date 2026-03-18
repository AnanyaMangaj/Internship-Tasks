#task14 :Student Marks Average Calculator
marks = [75, 44, 82, 88]
total = 0
count = 0
for m in marks:
    total = total + m
    count = count + 1

average = total / count
print("Average Marks:", average)
