"""
    Process guternbergdb.txt line by line, 
    saving language of origin for words to a file.

    Gutenberg dictionary from:
    http://www.gutenberg.org/ebooks/29765

"""

f = open("gutenbergdb.txt", "r")
lines = [x.replace("\n", "").replace("\r","") for x in f.readlines()]

for i, line in enumerate(lines):

    if "Etym:" in line:

        word = lines[i - 1].strip()
        language_of_origin = line.split("Etym:")[-1].replace("[","").split(".")[0].strip()
        print "%s, %s" % (word,language_of_origin)