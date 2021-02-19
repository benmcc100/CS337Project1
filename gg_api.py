# Project 1
# By:
# Mason Brachmann
# Ben McCormick
# Alberta Yoo
# Karim Noorani

### Imports ###

import json
import pandas as pd
import numpy as np
import sys
import spacy
nlp = spacy.load("en_core_web_sm", disable = ['textcat', 'lemmatizer'])
import nltk
from collections import Counter
import re
import regex
import os
import itertools
from nltk.corpus import stopwords

### Global Variables ###

key_word_award_mapping = {2013 : {
        "best screenplay - motion picture": ["best|mejor","screenplay", "motion", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best director - motion picture": ["best|mejor","director", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actress in a television series - comedy or musical" : ["best|mejor","actress", "comedy|musical", "television|tv|tele|series", "^(?!.*support).*$"],
        "best foreign language film" : ["best|mejor","foreign"],
        "best performance by an actor in a supporting role in a motion picture" : ["best|mejor","actor", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television" : ["best|mejor","actress", "support", "television|tv|tele|series|film", "mini|film"],
        "best motion picture - comedy or musical" : ["best|mejor","motion", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actress in a motion picture - comedy or musical" : ["best|mejor","actress", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best mini-series or motion picture made for television" : ["best|mejor","television|tv|series", "mini|film", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best original score - motion picture" : ["best|mejor","score", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        "best performance by an actress in a television series - drama" : ["best|mejor","actress", "television|tv|tele|series", "drama", "^(?!.*support).*$"],
        "best performance by an actress in a motion picture - drama" : ["best|mejor","actress", "drama", "^(?!.*support).*$", "^(?!.*series).*$"],
        "cecil b. demille award" : ["cecil|demille"],
        "best performance by an actor in a motion picture - comedy or musical" : ["best|mejor","actor", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"], 
        "best motion picture - drama" : ["best|mejor","motion", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television" : ["best|mejor","support", "actor", "television|tv|tele|series|film", "mini|film"],
        "best performance by an actress in a supporting role in a motion picture" : ["best|mejor","actress", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best television series - drama" : ["best|mejor","television|tv|series", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actor in a mini-series or motion picture made for television": ["best|mejor","actor", "television|tv|series", "mini|film" "^(?!.*support).*$"],
        "best performance by an actress in a mini-series or motion picture made for television" : ["best|mejor","actress", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        "best animated feature film" : ["best|mejor","animated", "film|feature"],
        "best original song - motion picture" : ["best|mejor","song", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        "best performance by an actor in a motion picture - drama" : ["best|mejor","actor", "drama", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best television series - comedy or musical" : ["best|mejor","television|tv|series", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actor in a television series - drama" : ["best|mejor","actor", "television|tv|series", "drama", "^(?!.*support).*$"],
        "best performance by an actor in a television series - comedy or musical" : ["best|mejor","actor", "television|tv|series", "comedy|musical", "^(?!.*support).*$"],
        "golden globe hosts": ["host", "golden globes"]
        }
        , 2015 : {"best screenplay - motion picture": ["best|mejor","screenplay", "motion", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best director - motion picture": ["best|mejor","director", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actress in a television series - comedy or musical" : ["best|mejor","actress", "comedy|musical", "television|tv|tele|series", "^(?!.*support).*$"],
        "best foreign language film" : ["best|mejor","foreign"],
        "best performance by an actor in a supporting role in a motion picture" : ["best|mejor","actor", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television" : ["best|mejor","actress", "support", "television|tv|tele|series|film", "mini|film"],
        "best motion picture - comedy or musical" : ["best|mejor","motion", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actress in a motion picture - comedy or musical" : ["best|mejor","actress", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best mini-series or motion picture made for television" : ["best|mejor","television|tv|series", "mini|film", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best original score - motion picture" : ["best|mejor","score", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        "best performance by an actress in a television series - drama" : ["best|mejor","actress", "television|tv|tele|series", "drama", "^(?!.*support).*$"],
        "best performance by an actress in a motion picture - drama" : ["best|mejor","actress", "drama", "^(?!.*support).*$", "^(?!.*series).*$"],
        "cecil b. demille award" : ["cecil|demille"],
        "best performance by an actor in a motion picture - comedy or musical" : ["best|mejor","actor", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"], 
        "best motion picture - drama" : ["best|mejor","motion", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television" : ["best|mejor","support", "actor", "television|tv|tele|series|film", "mini|film"],
        "best performance by an actress in a supporting role in a motion picture" : ["best|mejor","actress", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best television series - drama" : ["best|mejor","television|tv|series", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actor in a mini-series or motion picture made for television": ["best|mejor","actor", "television|tv|tele|series|film", "mini|film" "^(?!.*support).*$"],
        "best performance by an actress in a mini-series or motion picture made for television" : ["best|mejor","actress", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        "best animated feature film" : ["best|mejor","animated", "film|feature"],
        "best original song - motion picture" : ["best|mejor","song", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        "best performance by an actor in a motion picture - drama" : ["best|mejor","actor", "drama", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        "best television series - comedy or musical" : ["best|mejor","television|tv|series", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        "best performance by an actor in a television series - drama" : ["best|mejor","actor", "television|tv|series", "drama", "^(?!.*support).*$"],
        "best performance by an actor in a television series - comedy or musical" : ["best|mejor","actor", "television|tv|series", "comedy|musical", "^(?!.*support).*$"],
        "golden globe hosts": ["host", "golden globes"]},
        2018 : {
        'best motion picture - drama' : ["best|mejor","motion", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"], 
        'best motion picture - musical or comedy' : ["best|mejor","motion", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a motion picture - drama' : ["best|mejor","actress", "drama", "^(?!.*support).*$", "^(?!.*series).*$"],
        'best performance by an actor in a motion picture - drama' : ["best|mejor","actor", "drama", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a motion picture - musical or comedy' : ["best|mejor","actress", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actor in a motion picture - musical or comedy' : ["best|mejor","actor", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a supporting role in any motion picture' : ["best|mejor","actress", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actor in a supporting role in any motion picture' : ["best|mejor","actor", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best director - motion picture' : ["best|mejor","director", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best screenplay - motion picture' : ["best|mejor","screenplay", "motion", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best motion picture - animated' : ["best|mejor","animated", "film|feature"],
        'best motion picture - foreign language' : ["best|mejor","foreign"],
        'best original score - motion picture' : ["best|mejor","score", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        'best original song - motion picture' : ["best|mejor","song", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        'best television series - drama' : ["best|mejor","television|tv|series", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best television series - musical or comedy' : ["best|mejor","television|tv|series", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best television limited series or motion picture made for television' : ["best|mejor","television|tv|series", "mini|film", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best performance by an actress in a limited series or a motion picture made for television' : ["best|mejor","actress", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        'best performance by an actor in a limited series or a motion picture made for television' : ["best|mejor","actor", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        'best performance by an actress in a television series - drama' : ["best|mejor","actress", "television|tv|tele|series", "drama", "^(?!.*support).*$"],
        'best performance by an actor in a television series - drama' : ["best|mejor","actor", "television|tv|series", "drama", "^(?!.*support).*$"],
        'best performance by an actress in a television series - musical or comedy' : ["best|mejor","actress", "comedy|musical", "television|tv|tele|series", "^(?!.*support).*$"],
        'best performance by an actor in a television series - musical or comedy' : ["best|mejor","actor", "television|tv|series", "comedy|musical", "^(?!.*support).*$"],
        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television' : ["best|mejor","actress", "support", "television|tv|tele|series|film", "mini|film"],
        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television' : ["best|mejor","support", "actor", "television|tv|tele|series|film", "mini|film"],
        'cecil b. demille award' : ["cecil|demille"],
        "golden globe hosts": ["host", "golden globes"]},
        2019 : {
        'best motion picture - drama' : ["best|mejor","motion", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"], 
        'best motion picture - musical or comedy' : ["best|mejor","motion", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a motion picture - drama' : ["best|mejor","actress", "drama", "^(?!.*support).*$", "^(?!.*series).*$"],
        'best performance by an actor in a motion picture - drama' : ["best|mejor","actor", "drama", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a motion picture - musical or comedy' : ["best|mejor","actress", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actor in a motion picture - musical or comedy' : ["best|mejor","actor", "comedy|musical", "^(?!.*support).*$", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actress in a supporting role in any motion picture' : ["best|mejor","actress", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best performance by an actor in a supporting role in any motion picture' : ["best|mejor","actor", "support", "^(?!.*series).*$", "^(?!.*tv).*$"],
        'best director - motion picture' : ["best|mejor","director", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best screenplay - motion picture' : ["best|mejor","screenplay", "motion", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best motion picture - animated' : ["best|mejor","animated", "film|feature"],
        'best motion picture - foreign language' : ["best|mejor","foreign"],
        'best original score - motion picture' : ["best|mejor","score", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        'best original song - motion picture' : ["best|mejor","song", "^(?!.*actor).*$", "^(?!.*actress).*$", "^(?!.*by).*$"],
        'best television series - drama' : ["best|mejor","television|tv|series", "drama", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best television series - musical or comedy' : ["best|mejor","television|tv|series", "comedy|musical", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best television limited series or motion picture made for television' : ["best|mejor","television|tv|series", "mini|film", "^(?!.*actor).*$", "^(?!.*actress).*$"],
        'best performance by an actress in a limited series or a motion picture made for television' : ["best|mejor","actress", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        'best performance by an actor in a limited series or a motion picture made for television' : ["best|mejor","actor", "television|tv|tele|series|film", "mini|film", "^(?!.*support).*$"],
        'best performance by an actress in a television series - drama' : ["best|mejor","actress", "television|tv|tele|series", "drama", "^(?!.*support).*$"],
        'best performance by an actor in a television series - drama' : ["best|mejor","actor", "television|tv|series", "drama", "^(?!.*support).*$"],
        'best performance by an actress in a television series - musical or comedy' : ["best|mejor","actress", "comedy|musical", "television|tv|tele|series", "^(?!.*support).*$"],
        'best performance by an actor in a television series - musical or comedy' : ["best|mejor","actor", "television|tv|series", "comedy|musical", "^(?!.*support).*$"],
        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television' : ["best|mejor","actress", "support", "television|tv|tele|series|film", "mini|film"],
        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television' : ["best|mejor","support", "actor", "television|tv|tele|series|film", "mini|film"],
        'cecil b. demille award' : ["cecil|demille"],
        "golden globe hosts": ["host", "golden globes"]}}

awards = {2013 : ["best screenplay - motion picture", "best director - motion picture",
          "best performance by an actress in a television series - comedy or musical",
          "best foreign language film", "best performance by an actor in a supporting role in a motion picture",
          "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
          "best motion picture - comedy or musical",
          "best performance by an actress in a motion picture - comedy or musical",
          "best mini-series or motion picture made for television",
          "best original score - motion picture", "best performance by an actress in a television series - drama",
          "best performance by an actress in a motion picture - drama", "cecil b. demille award",
          "best performance by an actor in a motion picture - comedy or musical", "best motion picture - drama",
          "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
          "best performance by an actress in a supporting role in a motion picture", "best television series - drama",
          "best performance by an actor in a mini-series or motion picture made for television",
          "best performance by an actress in a mini-series or motion picture made for television",
          "best animated feature film", "best original song - motion picture",
          "best performance by an actor in a motion picture - drama", "best television series - comedy or musical",
          "best performance by an actor in a television series - drama",
          "best performance by an actor in a television series - comedy or musical"],
          2015 : ["best screenplay - motion picture", "best director - motion picture",
          "best performance by an actress in a television series - comedy or musical",
          "best foreign language film", "best performance by an actor in a supporting role in a motion picture",
          "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
          "best motion picture - comedy or musical",
          "best performance by an actress in a motion picture - comedy or musical",
          "best mini-series or motion picture made for television",
          "best original score - motion picture", "best performance by an actress in a television series - drama",
          "best performance by an actress in a motion picture - drama", "cecil b. demille award",
          "best performance by an actor in a motion picture - comedy or musical", "best motion picture - drama",
          "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
          "best performance by an actress in a supporting role in a motion picture", "best television series - drama",
          "best performance by an actor in a mini-series or motion picture made for television",
          "best performance by an actress in a mini-series or motion picture made for television",
          "best animated feature film", "best original song - motion picture",
          "best performance by an actor in a motion picture - drama", "best television series - comedy or musical",
          "best performance by an actor in a television series - drama",
          "best performance by an actor in a television series - comedy or musical"],
          2018 : ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award'],
          2019 : ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']
}

tokens_to_ignore = ["RT", "@", "golden globe", "goldenglobe", "globe", "globes", "golden", "hosts", "http", "#", "presentan", "globo", "wife", "y"]
potential_people_answers_to_ignore = ["hbo", "god", "motion", "picture", "winner", "award", "tv", "drama", "musical", "comedy", "actor", "actress", "movie", "honor", "star", "best", "mejor", "http", "@", "#", "fuck", "congrat", "fair", 'she', 'he', 'hooray' 'supporting', 'actor', 'actress', 'the', 'a', 'life', 'good', 'not', 'drinking', 'eating', 'and', 'hooray', 'nshowbiz', 'tmz', 'vanityfair', 'people', 'cnn', 'cbs', 'magazine', 'television', 'mejor', 'better', 'score', 'movie', 'film', 'picture', 'all', 'this', 'that', 'anyway', 'however', 'song', 'tune', 'music', 'drama', 'comedy', 'so', 'better', 'netflix', 'someone', 'mc', 'newz', 'season', 'should', 'fashion', 'has', 'how', 'oscar', 'grammy', 'oscars', 'oscars', 'drink', 'because', 'interesting', 'although', 'though', 'yay', 'congrats', "go"]
potential_answers_to_ignore = ["@", "golden globe", "goldenglobe", "globe", "globes", "golden", "hosts", "http", "#", "-", "motion picture", "tv", "movie", "mejor", "/", "congrats", "wins", "film", "best", "present", "foreign", "hollywood", "takes", "considering", "upset", "motionpicture", "wish", "should", "could", "won ", "nomintated"]
potential_awards_to_ignore = ["@", "golden globe", "goldenglobe", "globe", "globes", "golden", "hosts", "http", "#"]
### Helper Functions ###

###Sequence to Get Potential Answers
def getSolutionsFromPositions(doc, pos):
    #### This code will find take the doc and return a list of pos search that concates indices that are concurrent
    #### This idea was taken from https://stackoverflow.com/questions/63450423/how-to-find-proper-noun-using-spacy-nlp by user T. Jeanneau
    consecutives = []
    current = []
    for elt in pos:
        if len(current) == 0:
            current.append(elt)
        else:
            if current[-1] == elt - 1:
                current.append(elt)
            else:
                consecutives.append(current)
                current = [elt]
        if (len(current) != 0):
            consecutives.append(current)
    if ([doc[consecutive[0]:consecutive[-1]+1] for consecutive in consecutives] is None):
        return []
    else:
        return [doc[consecutive[0]:consecutive[-1]+1] for consecutive in consecutives]
    
### Checks if the tweet is relevant
def checkKeyWords(key_word_award_mapping, tweet):
    a = []
    for key, value in key_word_award_mapping.items():
        go = True
        for i in value:
            if not bool(re.search(i, tweet)):
                go = False
        if go:
            a.append(key)
    return a
    
### Updates Host Dictionary
def updateHostDictionary(doc, tweet, hosts, host_keys):  
    if (any(i for i in host_keys if(i in tweet))):
        pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]
        properPronouns = getSolutionsFromPositions(doc, pos)
        for h in (properPronouns):
            potential_host = h.text.lower()
            if not checkIfPerson(h):
                continue
            if (any(i for i in potential_answers_to_ignore if(i in potential_host))):
                continue
            if potential_host in hosts:
                hosts[potential_host] += 1
            else:
                hosts[potential_host] = 1
    return hosts

### Updates AwardNames Dictionary
def updateAwardNamesDictionary(doc, tweet, awardNames, award_name_keys):
    if (any(i for i in award_name_keys if(i in tweet)) and ("dress" not in tweet or "speech" not in tweet)):                 
        nouns = ["PROPN", "NOUN"]
        pos = [tok.i for tok in doc if ((tok.pos_ in nouns) or (tok.pos_ == "ADJ" and tok.text.lower() == "best") or (tok.text == "-")) and not any(i in tok.text.lower().split() for i in tokens_to_ignore)]
        words = getSolutionsFromPositions(doc, pos)
        for a in (words):
            potential_award_name = a.text.lower()
            if (potential_award_name[0:5] != "best "):
                continue
            if (len(potential_award_name.split()) < 4):
                continue
            if (any(i for i in potential_awards_to_ignore if(i in potential_award_name))):
                continue
            if "-" == potential_award_name.split()[-1]:
                continue
            if potential_award_name in awardNames:
                awardNames[potential_award_name] += 1
            else:
                awardNames[potential_award_name] = 1
    return awardNames

### Updates Winners Dictionary
def updateWinnersDictionary(doc, tweet, winners, matches, winner_keys):
    people_awards = ["act", "cecil", "director", "screen"]
    if (any(i for i in winner_keys if(i in tweet))):
        for award in matches:
            movies_tv_pos = ["PROPN", "NOUN", "ADJ", "AUX", "VERB"]
            if any(i for i in people_awards if(i in award)):
                pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))] 
            else:
                pos = [tok.i for tok in doc if ((tok.pos_ in movies_tv_pos or tok.text.lower() == "the" or tok.text.lower() == "of") and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]            
            properPronouns = getSolutionsFromPositions(doc, pos)
            for h in (properPronouns):
                potential_winner = h.text.lower()
                if any(i in potential_winner.split() for i in potential_people_answers_to_ignore) and checkIfPerson(h):
                    continue
                if (potential_winner in award):
                    continue
                if (any(i for i in people_awards if(i in award)) and (not checkIfPerson(h))):
                    continue
                if (any(i for i in potential_answers_to_ignore if(i in potential_winner))):
                    continue
                if (potential_winner[0:4] == "best"):
                    continue
                if award in winners:
                    if potential_winner in winners[award]:
                        winners[award][potential_winner] += 1
                    else:
                        winners[award][potential_winner] = 1
                else:
                    winners[award] = {}
                    winners[award][potential_winner] = 1
    return winners

### Updates Presenters Dictionary
def updatePresentersDictionary(doc, tweet, presenters, matches, presenter_keys):
    if (any(i for i in presenter_keys if(i in tweet))):
        for award in matches:
            pos = [tok.i for tok in doc if ((tok.pos_ == "PROPN" or tok.text.lower() == "will") and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]
            properPronouns = getSolutionsFromPositions(doc, pos)
            for h in (properPronouns):
                potential_presenter = h.text.lower()
                if not checkIfPerson(h):
                    continue
                if any(i in potential_presenter.split() for i in potential_people_answers_to_ignore):
                    continue
                if (potential_presenter in award):
                    continue
                if (potential_presenter == "best"):
                    continue
                if award in presenters:
                    if h.text.lower() in presenters[award]:
                        presenters[award][h.text.lower()] += 1
                    else:
                        presenters[award][h.text.lower()] = 1
                else:
                    presenters[award] = {}
                    presenters[award][h.text.lower()] = 1
    return presenters

### Updates Nominees Dictionary
def updateNomineesDictionary(doc, tweet, nominees, matches, nominees_keys):
    people_awards = ["act", "cecil", "director", "screen"]
    if (any(i for i in nominees_keys if(i in tweet))):
        for award in matches:
            movies_tv_pos = ["PROPN", "NOUN", "ADJ", "AUX", "VERB"]
            if ("director" in award or "act" in award or "cecil" in award):
                pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))] 
            else:
                pos = [tok.i for tok in doc if ((tok.pos_ in movies_tv_pos or tok.text.lower() == "the" or tok.text.lower() == "of") and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]            
            properPronouns = getSolutionsFromPositions(doc, pos)
            for h in (properPronouns):
                potential_nominee = h.text.lower()
                if (any(i for i in people_awards if(i in award)) and (not checkIfPerson(h))):
                    continue
                if any(i in potential_nominee.split() for i in potential_people_answers_to_ignore) and checkIfPerson(h):
                    continue
                if (potential_nominee in award):
                    continue
                if (any(i for i in potential_answers_to_ignore if(i in potential_nominee))):
                    continue
                if (potential_nominee[0:4] == "best" or potential_nominee[0:3] == "win") or potential_nominee == "the" or potential_nominee == "award" or "-" in potential_nominee:
                    continue
                if (potential_nominee[0:7] == "the win"):
                    continue
                if award in nominees:
                    if h.text.lower() in nominees[award]:
                        nominees[award][h.text.lower()] += 1
                    else:
                        nominees[award][h.text.lower()] = 1
                else:
                    nominees[award] = {}
                    nominees[award][h.text.lower()] = 1
    return nominees

### Updates Best Dressed Dictionary
def updateBestDressedDictionary(doc, tweet, best_dressed, best_dressed_keys):  
    pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]
    properPronouns = getSolutionsFromPositions(doc, pos)
    # Only want Phrases with 2-3 words (to represent a full name)
    for h in (properPronouns):
        potential_person = h.text.lower()
        if not checkIfPerson(h):
            continue
        if potential_person in best_dressed:
            best_dressed[potential_person] += 1
        else:
            best_dressed[potential_person] = 1
    return best_dressed

def updateWorstDressedDictionary(doc, tweet, worst_dressed, worst_dressed_keys):  
    pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]
    properPronouns = getSolutionsFromPositions(doc, pos)
    # Only want Phrases with 2-3 words (to represent a full name)
    for h in (properPronouns):
        potential_person = h.text.lower()
        if not checkIfPerson(h):
            continue
        if potential_person in worst_dressed:
            worst_dressed[potential_person] += 1
        else:
            worst_dressed[potential_person] = 1
    return worst_dressed

def updateFunnyDictionary(doc, tweet, funniest, funny_keys):
    if (any(i for i in funny_keys if(i in tweet))):
        pos = [tok.i for tok in doc if (tok.pos_ == "PROPN" and not any(i in tok.text.lower().split() for i in tokens_to_ignore))]
        properPronouns = getSolutionsFromPositions(doc, pos)
        # Only want Phrases with 2-3 words (to represent a full name)
        for h in (properPronouns):
            potential_person = h.text.lower()
            if not checkIfPerson(h):
                continue
            if "host" in tweet or "yay" in tweet:
                continue
            if potential_person in funniest:
                funniest[potential_person] += 1
            else:
                funniest[potential_person] = 1
    return funniest
 
### Updates the Final Answer Dictionary
def updateAnswerDictionary(year, hosts, awardNames, winners, presenters, nominees, best_dressed, worst_dressed, funniest, answer):
    #Hosts
    host_list = sorted(hosts.keys(), key=hosts.get, reverse=True)
    if (hosts != {}):
        p = np.percentile(list(hosts.values()),90)
        ans = []
        for key in host_list:
            if (hosts[key] > p):
                ans.append(key)
        if (len(ans) >= 2):
            answer["hosts"] = ans[:2]
        else:
            myList = []
            myList.append(ans[0])
            answer["hosts"] = myList

    #Winners
    winners.pop('golden globe hosts', None)
    for y in winners:
        answer["award_data"][y]["winner"] = max(winners[y], key=winners[y].get)

    #Awards
    # Removes the chance for award names selected to include the name of person/movie that won an award
    for y in answer["award_data"]:
        for k in awardNames.keys():
            if answer["award_data"][y]["winner"] in k:
                awardNames[k] = 0
    award_list = sorted(awardNames.keys(), key=awardNames.get, reverse = True)

    check_for_same_award_list = []
    ans = []
    for i in award_list:
        sort = ''.join(sorted(i))
        if sort not in check_for_same_award_list:
            ans.append(i)
            check_for_same_award_list.append(sort)
    answer["awards"] = ans[:27]


    #Presenters
    presenters.pop('golden globe hosts', None)
    
    for award in awards[year]:
        for x in presenters[award]:
            if answer["award_data"][award]["winner"] in x:
                presenters[award][x] = 0
    for y in presenters:      
        presenters_list = sorted(presenters[y].keys(), key=presenters[y].get, reverse=True)
        if (presenters[y] != {}):
            p = np.percentile(list(presenters[y].values()),1)
            ans = []
            for key in presenters_list:
                if (presenters[y][key] > p):
                    ans.append(key)
            if (len(ans) >= 2):
                answer["award_data"][y]["presenters"] = ans[:2]
            else:
                if ans == []:
                    myList = []
                    myList.append(presenters_list[0])
                    answer["award_data"][y]["presenters"] = myList
                else:
                    myList = []
                    myList.append(ans[0])
                    answer["award_data"][y]["presenters"] = myList

    #Nominees
    nominees.pop('golden globe hosts', None)
    for award in awards[year]:
        for x in nominees[award]:
            if answer["award_data"][award]["winner"] in x:
                nominees[award][x] = 0


    for y in nominees:
        nominees_list = sorted(nominees[y].keys(), key=nominees[y].get, reverse=True)
        answer["award_data"][y]["nominees"] = nominees_list[:4]


    #Best Dressed
    best_dressed_list = sorted(best_dressed.keys(), key=best_dressed.get, reverse=True)
    if (best_dressed_list != {}):
        answer["best_dressed"] = best_dressed_list[:5] 

    #Worst Dressed
    worst_dressed_list = sorted(worst_dressed.keys(), key=worst_dressed.get, reverse=True)
    if (worst_dressed_list != {}):
        answer["worst_dressed"] = worst_dressed_list[:5] 

    #Funniest
    funny_list = sorted(funniest.keys(), key=funniest.get, reverse=True)
    if (worst_dressed_list != {}):
        answer["funniest"] = funny_list[:5] 

    return answer


    
### Checks if the first word in a potential answer is a common name to determine if it is a person
def checkIfPerson(h):
    isPerson = False
    if ((len(h.text.split()) != 2) and (len(h.text.split()) != 3)):
        return False
    for ent in h.ents:
        if ent.label_ == "PERSON" or h.text.lower().split()[0] in names:
            isPerson = True
    return isPerson


### Make Human Readable File ###
def humanReadable(answer, year):
    string = "Year: " + str(year) + "\n"
    string += "Hosts: "
    for i in answer["hosts"]:
        string += i + ", "
    string = string[:-2]
    string += "\nAwards:" + "\n"
    for award in answer["awards"]:
        string += award + "\n"
    string += "\n"
    for award in answer["award_data"]:
        string += "Award: " + award
        string += "\n" + "Presenters: "
        for i in answer["award_data"][award]["presenters"]:
            string += i + ", "
        string = string[:-2]
        string += "\nNominees: "
        for i in answer["award_data"][award]["nominees"]:
            string += i + ", "
        string = string[:-2]
        string += "\nWinner: "
        string += answer["award_data"][award]["winner"]
        string += "\n\n"
    string += "\nBest Dressed on Red Carpet:\n"
    string += str(answer["best_dressed"])
    string += "\nWorst Dressed on Red Carpet:\n"
    string += str(answer["worst_dressed"])
    string += "\nFunniest Speakers:\n"
    string += str(answer["funniest"])

    f = open("%s_HumanReadableAnswers.txt" % year, "w", encoding="utf-8")
    f.write(string)
    f.close()

### Required Functions ###

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    hosts = answer["hosts"]
    return hosts

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    awards = answer["awards"]
    return awards

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    nominees = {}
    for i in awards[int(year)]:
        nominees[i] = answer["award_data"][i]["nominees"]
    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    winners = {}
    for i in awards[int(year)]:
        winners[i] = answer["award_data"][i]["winner"]
    return winners

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    presenters = {}
    for i in awards[int(year)]:
        presenters[i] = answer["award_data"][i]["presenters"]
    return presenters

def get_best_dressed(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    best_dressed_people = answer["best_dressed"]
    return best_dressed_people

def get_worst_dressed(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    worst_dressed_people = answer["worst_dressed"]
    return worst_dressed_people

def get_funniest_speakers(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    json_file = open("%sanswers.json" % year)
    answer = json.load(json_file)
    json_file.close()
    funniest_people = answer["funniest"]
    return funniest_people

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    print("Pre-ceremony starting")
    ## Set Up JSON
    global answer
    global years
    global names
    for year in years:
        answer = {}
        answer["hosts"] = []
        answer["awards"] = {}
        answer["award_data"] = {}
        answer["best_dressed"] = []
        for i in awards[year]:
            answer["award_data"][i] = {"nominees": [], "presenters": [], "winner":""}

        ## Make Dictionaries for hosts, nominees, presenters, and winners
        hosts = {}
        awardNames = {}
        winners = {}
        presenters = {}
        nominees = {}
        best_dressed = {}
        worst_dressed = {}
        funniest = {}
        ## Cycle Through Tweets
        tweets = pd.read_json("gg" + str(year) + ".json")
        for x, col in tweets.iterrows():
            tweet = col["text"]
            if "RT" in tweet: #Prevents getting the same tweet multiple times
                continue
            #This if statement is so tweets that dont fit any of these are not looked at at all
            matches = checkKeyWords(key_word_award_mapping[year], tweet.lower())
            if matches != []:
                #if 'golden globe hosts' in matches:
                #if ((("won" in tweet.lower() or "congrat" in tweet.lower() or "winner" in tweet.lower() or "goes to" in tweet.lower()) and ("best" in tweet.lower() or "cecil" in tweet.lower())) or ("present" in tweet or "announc" in tweet or "award" in tweet or "give" in tweet)):                    
                winner_keys = ["won", "congrat", "winner", "goes to"]
                presenter_keys = ["present", "announc", "award", "give"]
                award_name_keys = ["best"]
                host_keys = ["host", "golden globes"]
                nominees_keys = ["nom", "hope", "should", "wish", "chance", "could", "go", "love", "beat"]
                funny_keys = ["funny", "funniest", "hilarious", "haha", "joke", "laugh"]
                if ((any(i for i in winner_keys if(i in tweet.lower()))) or (any(i for i in presenter_keys if(i in tweet.lower()))) or (any(i for i in host_keys if(i in tweet.lower()))) or (any(i for i in nominees_keys if(i in tweet.lower()))) or (any(i for i in funny_keys if(i in tweet.lower())))):
                    doc = nlp(tweet)
                else:
                    continue
            elif ("http://t.co" in tweet):
                # see if image is being commented on with adjectives befitting "best dressed" classification
                best_dressed_keys = ["suit", "dress", "red carpet"]
                best_dressed_keys2 = ["best", "amazing"]
                worst_dressed_keys = ["worst", "ugly", "horrible", "terrible"]
                if((any(i for i in best_dressed_keys if(i in tweet.lower()))) and (any(i for i in best_dressed_keys2 if(i in tweet.lower())))):
                    doc = nlp(tweet)
                    best_dressed = updateBestDressedDictionary(doc, tweet.lower(), best_dressed, best_dressed_keys)
                elif ((any(i for i in best_dressed_keys if(i in tweet.lower()))) and (any(i for i in worst_dressed_keys if(i in tweet.lower())))):
                    doc = nlp(tweet)
                    worst_dressed = updateWorstDressedDictionary(doc, tweet.lower(), worst_dressed, worst_dressed_keys)
                continue
            else:
                continue
                
            tweet = tweet.lower()
            ##Check if Host
            hosts = updateHostDictionary(doc, tweet, hosts, host_keys)
            #Check if Award Name
            awardNames = updateAwardNamesDictionary(doc, tweet, awardNames, award_name_keys)
            #Check if Award Winner
            winners = updateWinnersDictionary(doc, tweet, winners, matches, winner_keys)
            #Check for Presenters
            presenters = updatePresentersDictionary(doc, tweet, presenters, matches, presenter_keys)
            #Check for Nominees
            nominees = updateNomineesDictionary(doc, tweet, nominees, matches, nominees_keys)
            #Check for Funniest Peopel
            funniest = updateFunnyDictionary(doc, tweet, funniest, funny_keys)
        #Fill in final JSON
        answer = updateAnswerDictionary(year, hosts, awardNames, winners, presenters, nominees, best_dressed, worst_dressed, funniest, answer)

        with open("%sanswers.json" % year, "w") as outfile:
            json.dump(answer, outfile)
        humanReadable(answer, year)

    print("Pre-ceremony processing complete.")
    return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    global answer
    answer = {}
    global years
    years = [2013,2015]
    global names
    with open('names.txt',"r") as f:
        names = f.read()
    names = names.lower().split("\n")
    answer = pre_ceremony()   
    return
if __name__ == '__main__':
    main()