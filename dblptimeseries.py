import dblp

# the dblp package is at https://github.com/scholrly/dblp-python
# clone that, then run `python setup.py install`

# PSUEDOCODE (should be easy. trickiest part will be filtering out wrong authors)
# author-names = the author names in our hiring network dataset
#
# authors = {}
# for author in author-names:
#     authors[author] = dblp.search(author)
#     filter out wrong authors (synonyms/homonyms) to get new authors dict
#     authors = filter-authors(authors, author-names, ?, ?, ...)
#     get author-pubs, a dictionary from author names to their list of publication dictionaries (relevant keys: year, ee (the DOI))
#     transform author-pubs to time-series, Year -> [AuthorName * DOI * Publication]
