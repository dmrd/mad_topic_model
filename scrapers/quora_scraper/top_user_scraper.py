"""
    Generates top_users.txt 
"""

import urllib
from bs4 import BeautifulSoup

usernames = []

# quoratop
f = urllib.urlopen("http://www.quoratop.com/")
html = f.read()
soup = BeautifulSoup(html)
usernames.extend([link.get('href').split("/")[-1] for link in soup.find_all("a", {"class":"name"})])

# http://www.quora.com/Recommended-Users-on-Quora/Whom-should-one-follow-on-Quora-and-why
# http://www.quora.com/Quora-Usage-Data-and-Analysis/What-Quora-users-have-the-most-followers-1
dumps = ["quora_dump1.html", "quora_dump2.html"]

for file_name in dumps:
    f = open(file_name, "r")
    html = f.read()
    soup = BeautifulSoup(html)

    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            usernames.append(href.split("/")[-1])

usernames = list(set(usernames))

with open("top_users.txt", "w") as f:
    f.write("\n".join(usernames))