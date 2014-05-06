# Scrapy settings for course_scraper project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'etym_scraper'

SPIDER_MODULES = ['etym_scraper.spiders']
NEWSPIDER_MODULE = 'etym_scraper.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'etym_scraper (+http://www.yourdomain.com)'
