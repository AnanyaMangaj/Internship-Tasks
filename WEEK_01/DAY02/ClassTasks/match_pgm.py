#program using match-case for months

month = input("Enter month (jan, feb, mar ... dec): ").lower()

match month:
    case "jan":
        print("January - First month of the year")
    case "feb":
        print("February - Second month of the year")
    case "mar":
        print("March - Third month of the year")
    case "apr":
        print("April - Fourth month of the year")
    case "may":
        print("May - Fifth month of the year")
    case "jun":
        print("June - Sixth month of the year")
    case "jul":
        print("July - Seventh month of the year")
    case "aug":
        print("August - Eighth month of the year")
    case "sep":
        print("September - Ninth month of the year")
    case "oct":
        print("October - Tenth month of the year")
    case "nov":
        print("November - Eleventh month of the year")
    case "dec":
        print("December - Last month of the year")
    case _:
        print("Invalid month entered")
