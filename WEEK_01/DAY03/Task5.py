#Task 05: file data counter
def count_file_data(filename):
    line_count = 0
    word_count = 0
    char_count = 0

    file = open(filename, "r")

    for line in file:
        line_count = line_count + 1
        words = line.split()
        word_count = word_count + len(words)
        char_count = char_count + len(line)

    file.close()
    print("Lines:", line_count)
    print("Words:", word_count)
    print("Characters:", char_count)
file_name = input("Enter file name: ")
count_file_data(file_name)
