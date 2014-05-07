"""
    Scrapes Quora for users' answers

    Usage:

        python scraper.py [users file]
"""

import sys
import time
import urllib
import csv
from splinter import Browser

usernames = [line.replace("\n", "") for line in open(sys.argv[1], "r").readlines()]
num_usernames = len(usernames)

browser = Browser("phantomjs")


for username_counter, username in enumerate(usernames):

    posts = [] # list of tuples (username, answer)

    try:
        browser.visit("http://www.quora.com/%s/answers" % username)
    except:
        print "Failed on %s" % username
        continue


    # get rid of the login overlay
    try:
        browser.find_by_css(".cancel").first.click()
        time.sleep(1)
    except:
        pass


    # scroll to bottom several times
    for i in range(5):
        browser.evaluate_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    for element in browser.find_by_css(".inline_editor_value"):
        try:
            text = element.find_by_css("div").first.find_by_css("div").first.text
            text_ascii = text.encode('ascii', 'replace')
            posts.append((username,text_ascii))
        except:
            pass


    with open("output.csv", "a") as f:
        csv_out = csv.writer(f)
        csv_out.writerows(posts)

    print "Completed %s of %s: %s" % (username_counter, num_usernames, username)