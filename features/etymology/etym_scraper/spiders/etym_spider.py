from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from etym_scraper.items import Entry


class EtymSpider(CrawlSpider):
    name = "etym_scraper"
    allowed_domains = ["etymonline.com"]
    start_urls = [
        "http://www.etymonline.com/index.php"
    ]

    rules = (
        Rule(SgmlLinkExtractor(allow=('index.php')), callback='parse_details'),
    )

    def parse_details(self, response):
        sel = Selector(response)

        all_dt = sel.xpath('//dt')

        results = []

        for dt in all_dt:
            entry = Entry()
            entry['word'] = dt.xpath('.//a[1]/text()').extract()
            entry['etym'] = dt.xpath(
                './following-sibling::dd/text()').extract()

            results.append(entry)

        return results
