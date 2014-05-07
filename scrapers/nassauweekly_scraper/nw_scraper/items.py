# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

class Article(Item):
    url = Field()
    date = Field()
    author = Field()
    title = Field()
    body = Field()