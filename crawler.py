#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from load import get_soup
import re
from cleaner import remove_non_text_tags, extract_text

wp_url_matcher = re.compile(r'^/wiki/(.*)')


def crawl(to_load_list):
    loaded_articles = os.listdir("articles")
    for _ in range(1000):
        print("To Load List length: " + str(len(to_load_list)))
        if not to_load_list:
            break
        to_load_url = to_load_list.pop()
        soup = get_soup("https://de.wikipedia.org/wiki/" + to_load_url)
        content = soup.find("div", {"id": "mw-content-text"})
        for a in content.find_all("a"):
            match = wp_url_matcher.match(a["href"])
            if match and match.group(1) not in loaded_articles and \
                    not (match.group(1).startswith("Wikipedia:") or
                         match.group(1).startswith("Spezial:") or
                         match.group(1).startswith("Datei:") or
                         match.group(1).startswith("Hilfe:") or
                         match.group(1).count("/") > 0):
                to_load_list.append(match.group(1))
        remove_non_text_tags(soup)
        with open(os.path.join("articles", to_load_url), 'w') as f:
            f.write(extract_text(soup))
            loaded_articles.append(to_load_url)

if __name__ == "__main__":
    crawl(["Heinrich_von_Kleist"])
