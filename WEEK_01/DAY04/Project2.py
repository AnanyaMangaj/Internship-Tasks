#Project2 : word counter tool
file = open("summary.txt","w")
file.write("Python is easy\nFile handling project\nDay 4 learning")
file.close()

print("summary.txt created")

def word_counter(filename):
    try:
        file = open(filename, "r")
        text = file.read()
        file.close()
        lines = text.split("\n")
        words = text.split()
        print("Lines:", len(lines))
        print("Words:", len(words))
        print("Characters:", len(text))
    except:
        print("File not found")

word_counter("summary.txt")
