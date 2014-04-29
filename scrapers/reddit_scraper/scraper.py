import sys
import praw

r = praw.Reddit(user_agent="Reddit Top Comments Scraper")

user_name = sys.argv[1]

user = r.get_redditor(user_name)

for comment in user.get_comments(sort="hot"): 

    print "==="
    print comment
    print "==="