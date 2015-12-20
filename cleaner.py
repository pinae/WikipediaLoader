#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from bs4 import Comment


def remove_double_newlines(text):
    pattern = re.compile(r'\n\n+')
    return pattern.sub("\n", text)


def remove_non_text_tags(soup):
    # remove the edit links
    for tag in soup.find_all("span", {"class": "mw-editsection"}):
        tag.decompose()
    # Remove the table of contents
    toc = soup.find("div", {"id": "toc"})
    if toc:
        toc.replaceWith("")
    # remove thumbnail images and their description
    for tag in soup.find_all("div", {"class": "thumbinner"}):
        tag.decompose()
    # remove floating boxes
    for tag in soup.find_all("div", {"class": "float-right"}):
        tag.decompose()
    # remove catlinks
    catlinks = soup.find("div", {"id": "catlinks"})
    if catlinks:
        catlinks.replaceWith("")
    # remove normdaten
    normdaten = soup.find("div", {"id": "normdaten"})
    if normdaten:
        normdaten.replaceWith("")
    # remove nav frames
    for tag in soup.find_all("div", {"class": "NavFrame"}):
        tag.decompose()
    # remove printfooters
    for tag in soup.find_all("div", {"class": "printfooter"}):
        tag.decompose()
    # remove mw-jump
    for tag in soup.find_all("div", {"class": "mw-jump"}):
        tag.decompose()
    # remove hidden tags
    for tag in soup.select('[style~="display:none"]'):
        tag.decompose()
    # remove geo tags
    for tag in soup.find_all("span", {"class": "geo"}):
        tag.decompose()
    for tag in soup.find_all("span", {"class": "coordinates"}):
        tag.decompose()
    # remove tables
    for tag in soup.find_all("table"):
        tag.decompose()
    # remove reference marker
    for tag in soup.find_all("sup", {"class": "reference"}):
        tag.replaceWith("")
    # remove all comments
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.replaceWith("")
    # remove hauptartikel references
    for tag in soup.find_all("div", {"class": "hauptartikel"}):
        tag.decompose()


def extract_text(soup):
    heading = soup.find("h1", {"id": "firstHeading"})
    text = remove_double_newlines(heading.text) + "\n"
    p = ["", ""]
    section_wanted = True
    content = soup.find("div", {"id": "mw-content-text"})
    for tag in content:
        if tag.name in ["h1", "h2", "h3", "h4", "h5"]:
            if section_wanted:
                if p[0]:
                    text += "\n\n" + p[0] + "\n"
                text += remove_double_newlines(p[1])
            p = ["", ""]
            section_wanted = not tag.text.strip() in [
                "Literatur",
                "Quellen",
                "Einzelnachweise",
                "Siehe auch",
                "Filme",
                "Musik",
                "Weblinks"]
            if section_wanted:
                p[0] = remove_double_newlines(tag.text)
        else:
            if section_wanted:
                if tag.name is None:
                    p[1] += str(tag)
                else:
                    if tag.find("img", {"class": "tex"}):  # Removing sections with formulas
                        section_wanted = False
                    else:
                        p[1] += tag.text
    if section_wanted:
        text += "\n\n" + p[0] + "\n" + remove_double_newlines(p[1])
    return text
