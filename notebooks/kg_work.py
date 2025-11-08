import json
import sys
import pandas as pd
import collections 
import os
import numpy as np
from itertools import chain
from itertools import combinations
sys.path.insert(0, '..')
from src.experiment_utils.helper_classes import token, span, repository
from src.d02_corpus_statistics.corpus import Corpus
import types
from owlready2 import sync_reasoner

cwd = os.getcwd()
pol_dir = cwd+"/../src/d01_data"

pol_df = pd.read_pickle(pol_dir+"/preprocessed_dataframe.pkl")[["Policy","Text","Tokens","Curation"]]
meta_df = pd.read_csv(pol_dir+"/EU_metadata.csv", delimiter=";")

from owlready2 import *

owl_path = cwd+"/auxil/ontology.owl"
onto = get_ontology(owl_path).load()

with onto:
    # classes
    #policy structure
    class Policy(Thing): 
        pass
    class Chapter(Thing): 
        pass
    class Section(Thing): 
        pass
    class Article(Thing): 
        pass
    #spans
    class Layer(Thing): 
        pass
    class Feature(Thing): 
        pass
    class Tag(Thing): 
        pass
    class Span(Thing): 
        pass

    # object properties
    #structure
    # should I change these to "part of" and "type of" instead of specific relationships??
    class hasPart(ObjectProperty):
        domain = [Policy, Chapter, Section, Article]
        range = [Chapter, Section, Article, Span]
    class partOf(ObjectProperty):
        domain = [Chapter, Section, Article, Span]
        range = [Policy, Chapter, Section, Article]
    class hasChapter(hasPart):
        domain = [Policy]
        range = [Chapter]
    class hasSection(hasPart):
        domain = [Chapter]
        range = [Section]
    class hasArticle(hasPart):
        domain = [Section]
        range = [Article]
    class hasSpan(hasPart):
        domain = [Article]
        range = [Span]
    class isChapterOf(partOf):
        domain = [Chapter]
        range = [Policy]
        inverse_property = hasChapter
    class isSectionOf(partOf):
        domain = [Section]
        range = [Chapter]
        inverse_property = hasSection
    class isArticleOf(partOf):
        domain = [Article]
        range = [Section]
        inverse_property = hasArticle
    class isSpanOf(partOf):
        domain = [Span]
        range = [Article]
        inverse_property = hasSpan
    #spans
    class inTag(ObjectProperty):
        domain = [Span]
        range = [Tag]
    class hasSpan(ObjectProperty):
        domain = [Tag]
        range = [Span]
        inverse_property = inTag
    ### ??????????????
    ### do i keep these here or specify strict elsewhere?
    ### since tags always have the same feature which always have the same layer
    class inLayer(ObjectProperty):
        domain = [Feature]
        range = [Layer]
    class inFeature(ObjectProperty):
        domain = [Tag]
        range = [Feature]
    class hasFeature(ObjectProperty):
        domain = [Layer]
        range = [Feature]
        inverse_property = inLayer
    class hasTag(ObjectProperty):
        domain = [Feature]
        range = [Tag]
        inverse_property = inFeature
    
    # data properties
    #policy structure
    class policy_code(DataProperty):
        domain = [Policy]
        range = [str]
    class chapter_num(DataProperty):
        domain = [Chapter]
        range = [str]
    class section_num(DataProperty):
        domain = [Section]
        range = [str]
    class article_num(DataProperty):
        domain = [Article]
        range = [str]
    #spans
    class layer_name(DataProperty):
        domain = [Layer]
        range = [str]
    class feature_name(DataProperty):
        domain = [Feature]
        range = [str]
    class tag_name(DataProperty):
        domain = [Tag]
        range = [str]
    class span_id(DataProperty):
        domain = [Span]
        range = [str]
    class span_text(DataProperty):
        domain = [Span]
        range = [str]

policies = {}
chapters = {}
sections = {}
articles = {}

for ind in pol_df.index:
    deets = ind.split("_")
    policy_code = "_".join(deets[:2])
    #ignore whereas and front bits for now
    if deets[2] == "Whereas" or deets[2] == "front":
        continue
    chapter_num = deets[5]
    section_num = deets[7]
    article_num = deets[9]
    # Policy
    policy = onto[policy_code]
    if not policy:
        policy = onto.Policy(policy_code) #unique Policy instance with name of policy_code
        policy.policy_code = [policy_code] #sets the Policy instance's policy_code to the policy_code
        policies[policy_code] = policy #stores the Policy instance in the dictionary
    # Chapter
    chapter_key = f"{policy_code}_Chapter_{chapter_num}" #makes unique chapter key
    chapter = onto[chapter_key]
    if not chapter:
        chapter = onto.Chapter(chapter_key)
        chapter.chapter_num = [chapter_num]
        chapter.isChapterOf.append(policy)
        chapters[chapter_key] = chapter
    # Section
    section_key = f"{policy_code}_Chapter_{chapter_num}_Section_{section_num}"
    section = onto[section_key]
    if not section:
        section = onto.Section(section_key)
        section.section_num = [section_num]
        section.isSectionOf.append(chapter)
        sections[section_key] = section
    # Article
    article_key = f"{policy_code}_Chapter_{chapter_num}_Section_{section_num}_Article_{article_num}"
    article = onto[article_key]
    if not article:
        article = onto.Article(article_key)
        article.article_num = [article_num]
        article.isArticleOf.append(section)
        articles[article_key] = article

no_tag_lst = []
for ind in pol_df.index:
    deets = ind.split("_")
    policy_code = "_".join(deets[:2])
    #ignore whereas and front bits for now
    if deets[2] == "Whereas" or deets[2] == "front":
        continue
    chapter_num = deets[5]
    section_num = deets[7]
    article_num = deets[9]
    article_key = f"{policy_code}_Chapter_{chapter_num}_Section_{section_num}_Article_{article_num}"

    article = onto[article_key]
    for spanobj in pol_df.loc[ind, "Curation"]:
        span_id = f"{article_key}_{spanobj.span_id}"
        if not spanobj.tag:
            no_tag_lst.append((ind, spanobj.span_id))
            continue
        if spanobj.feature == "Technologyandapplicationspecificity":
            continue
        span = onto.Span(span_id+"_span")
        span.span_id = [span_id]
        span.span_text = [spanobj.text]
        # tag
        tag = onto[spanobj.tag]
        if not tag:
            tag = onto.Tag(spanobj.tag)
            tag.tag_name = [spanobj.tag]
        # can i assert feature/layer inheritance globally? should i?
        feature = onto[spanobj.feature]
        if not feature:
            feature = onto.Feature(spanobj.feature)
            feature.feature_name = [spanobj.feature]
        layer = onto[spanobj.layer]
        if not layer:
            layer = onto.Layer(spanobj.layer)
            layer.layer_name = [spanobj.layer]
        span.inTag = [tag]
        tag.inFeature = [feature]
        feature.inLayer = [layer]
        #tag.taggedSpan.append(span)
        span.isSpanOf = [article]
        #article.hasSpan.append(span)
    articles[article_key] = article

sync_reasoner(infer_property_values=True)
onto.save(file=f"{cwd}/auxil/policy_kg_populated.owl", format="rdfxml")