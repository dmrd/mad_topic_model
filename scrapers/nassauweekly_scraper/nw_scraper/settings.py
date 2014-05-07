# Scrapy settings for course_scraper project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'nw_scraper'

SPIDER_MODULES = ['nw_scraper.spiders']
NEWSPIDER_MODULE = 'nw_scraper.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'nw_scraper (+http://www.yourdomain.com)'

DEPTH_LIMIT = 3
