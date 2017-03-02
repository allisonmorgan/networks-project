# -*- coding: utf-8 -*-

import gzip
import re
import sys

from bs4 import BeautifulSoup

# Various article attirbutes
author_string = 'author'
title_string = 'title'
year_string = 'year'
journal_string = 'journal'
doi_string = 'ee'
url_string = 'url'

if __name__ == "__main__":
    keyword = sys.argv[1]
    matches = int(sys.argv[2])

    count_dois = 0
    paper = {}

    with gzip.open("dblp.gz", "r") as file:
        for i, line in enumerate(file):
            soup = BeautifulSoup(line, 'xml')

            authors = soup.find(author_string)
            if authors is not None:
                for author in authors:
                    paper[author_string] = author

            titles = soup.find(title_string)
            if titles is not None:
                # There can be tags within a title tag (<i>), remove those tags 
                # and append their text
                paper_title = ""
                for i, title in enumerate(titles):
                    # Why do titles always end with a period?
                    title_part = unicode(title).rstrip(".")
                    title_part = re.sub("<i>", "", re.sub("</i>", "", title_part))
                    paper_title = paper_title + title_part
                
                paper[title_string] = paper_title

            years = soup.find(year_string)
            if years is not None:
                for year in years:
                    paper[year_string] = year

            journals = soup.find(journal_string)
            if journals is not None:
                for journal in journals:
                    paper[journal_string] = journal

            urls = soup.find(url_string)
            if urls is not None:
                for url in urls:
                    paper[url_string] = url

            dois = soup.find(doi_string)
            if dois is not None:
                for doi in dois:
                    paper[doi_string] = doi
                    
                    if paper.has_key(title_string) and paper[title_string].lower().count(keyword):
                    	count_dois = count_dois + 1

                        print(paper)
                        print(">>>>>>")

                    paper = {}


            if count_dois == matches:
                break
