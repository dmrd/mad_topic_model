"""
    Process guternbergdb.txt line by line,
    saving language of origin for words to a file.

    Gutenberg dictionary from:
    http://www.gutenberg.org/ebooks/29765

"""

etymologies = ['GR.', 'NL.', 'L.', 'LL.', 'OF.', 'F.', 'OE.', 'AS.']

f = open("gutenbergdb.txt", "r")
lines = [x.replace("\n", "").replace("\r", "") for x in f.readlines()]

for i, line in enumerate(lines):

    if "Etym:" in line:

        word = lines[i - 1].strip().split(',')[0]

        bad = set('0123456789$,*\"\'().:;')
        if any((c in bad) for c in word) or not word:
            word = lines[i - 2].strip().split(',')[0]
        if any((c in bad) for c in word) or not word:
            continue

        line = line.split("Defn:")[0]
        language_of_origin = line.split("Etym:")[-1].replace("[", "").strip()
        for et in etymologies:
            if et in language_of_origin:
                for word in word.split(";"):
                    print "%s, %s" % (word.strip().lower(), et)
                break
