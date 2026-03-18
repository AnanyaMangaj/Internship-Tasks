#Mini project:Performance rating system(conditional statement)
#scenerio:assign performance rating
name = input("Enter employee name: ")
score = int(input("Enter performance score (0 - 100): "))

print("\nEvaluating Performance...\n")

if score >= 90:
    rating = "Excellent"

elif score >= 75:
    rating = "Very Good"

elif score >= 60:
    rating = "Good"

elif score >= 40:
    rating = "Average"

else:
    rating = "Poor"

print("Employee Name:", name)
print("Performance Score:", score)
print("Performance Rating:", rating)
