##########################################
# Cleans a project gutenberg text file
# Removes header/footer
# See http://cs.brown.edu/courses/csci0931/2014/3-synthesis/HW3-1/HW3-1.py
##########################################
import sys
import re

GUTENBERG_DIVIDER = r'(\*)+\s*(START|END).*PROJECT GUTENBERG.*$'
GUTENBERG_PARA_PAT = r'\n{2,}[^\n]*project\s+gutenberg([^\n]+\n)+'
PRODUCED_PARA_PAT = r'\n+produced\s+by([^\n]+\n)+'

try:
    filename = sys.argv[1]
except:
    print("Usage: {} filename".format(sys.argv[0]))
    exit()

with open(filename) as f:
    text = f.read()

# Convert to standard line breaks. Windows uses slightly different line breaks
# than other operating systems, so if we aren't sure how the file was saved,
# this is a way to standardize it.
text = re.sub(r'\r?\n', '\n', text)

# Strip the header.
header_match = re.search(GUTENBERG_DIVIDER, text, re.MULTILINE)
text = text[header_match.end()+1 : len(text)]

# Strip the footer.
footer_match = re.search(GUTENBERG_DIVIDER, text, re.MULTILINE)
text = text[0 : footer_match.start()]

# Remove any paragraphs that mention Project Gutenberg.
text = re.sub(GUTENBERG_PARA_PAT, '', text, flags=re.IGNORECASE)

# Remove who produced this transcription
text = re.sub(PRODUCED_PARA_PAT, '', text, flags=re.IGNORECASE)

print(text.strip())
