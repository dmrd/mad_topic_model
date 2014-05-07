from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from nw_scraper.items import Article


class EtymSpider(CrawlSpider):
    name = "nw_scraper"
    allowed_domains = ["www.nassauweekly.com"]
    start_urls = [
        "http://www.nassauweekly.com/"
    ]

    rules = (
        Rule(SgmlLinkExtractor(allow=()), callback='parse_details', follow=True),
    )

    def parse_details(self, response):
        sel = Selector(response)

        article = Article()

        article['url'] = response.url
        article['title'] = sel.css('div.post > div.post-meta > h1::text').extract()
        article['date'] = sel.css('div.post > div.post-meta > span:nth-child(4)::text').extract()
        article['author'] = sel.css('div.post > div.post-meta > span.post-author > a:nth-child(1)::text').extract()
        article['body'] = sel.css('div.post > div.post-content > p::text').extract()


        if article['author'] and article['body']:
            return [article]
        else:
            return []
