"""
    Generates top_users.txt 
"""

import urllib
from bs4 import BeautifulSoup

f = urllib.urlopen("http://www.quoratop.com/")
html = f.read()
soup = BeautifulSoup(html)

usernames = [link.get('href').split("/")[-1] for link in soup.find_all("a", {"class":"name"})]

with open("top_users.txt", "w") as f:
    f.write("\n".join(usernames))