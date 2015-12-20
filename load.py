#!/usr/bin/python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from cleaner import remove_non_text_tags, extract_text
from encoder import encode, decode


def get_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.text)


if __name__ == "__main__":
    soup = get_soup("https://de.wikipedia.org/wiki/Speex")
    remove_non_text_tags(soup)
    print(decode(encode(extract_text(soup))))
