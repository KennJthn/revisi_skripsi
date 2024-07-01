#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os



from latex_utils import latex_from_dataframe


# In[ ]:


DIR = os.getcwd()

# Inputs
ARTICLES_INPUT_DIR_PATH = os.path.join(DIR, 'coliee_documents', 'civil_code_collection.txt')
TRAIN_QUERIES_DIR = os.path.join(DIR, 'train')
TEST_QUERIES_DIR = os.path.join(DIR, 'test')

# Outputs
ARTICLES_OUTPUT_PATH = os.path.join(DIR, 'coliee_documents', 'civil_code_collection.csv')
TRAIN_QUERIES_OUPTUT_PATH = os.path.join(TRAIN_QUERIES_DIR, 'train_query.csv')
TEST_QUERIES_OUPTUT_PATH = os.path.join(TEST_QUERIES_DIR, 'test_query.csv')

OUTPUT_ENCODER_DIR = os.path.join(DIR, 'raw_text_for_encoder')
ARTICLES_OUTPUT_ENCODER_PATH = os.path.join(OUTPUT_ENCODER_DIR, 'raw_civil_code_collection.csv')
QUERIES_OUTPUT_ENCODER_PATH = os.path.join(OUTPUT_ENCODER_DIR, 'raw_queries.csv')


# # Parse Documents

# In[ ]:
import re

def is_part(line: str) -> bool:
    return re.match(r'Part [IVX]+', line) is not None

def is_chapter(line: str) -> bool:
    return re.match(r'Chapter [IVX]+', line) is not None

def is_section(line: str) -> bool:
    return re.match(r'Section [\d-]+', line) is not None

def is_subsection(line: str) -> bool:
    return re.match(r'Subsection [\d-]+', line) is not None

def is_subsubsection(line: str) -> bool:
    return re.match(r'^\(.*\)$', line) is not None

def is_article(line: str) -> bool:
    return re.match(r'Article [\d-]+', line) is not None

def is_clause(line: str) -> bool:
    return re.match(r'^\(\S*\) .+', line) is not None


# In[ ]:


def parse_lines(lines: str) -> list[tuple]:
    articles = []
    part, chapter, sect, subsect, subsubsect, art, art_code = '', '', '', '', '', '', ''

    for line in lines:
        line = line.strip()
        tokens = line.split()

        if is_clause(line):
            art += '\n' + line
            continue

        if art != '':
            articles.append((part, chapter, sect, subsect, subsubsect, art, art_code))

        if is_part(line):
            part = ' '.join(tokens[2:])
            chapter, sect, subsect, subsubsect, art, art_code = '', '', '', '', '', ''
        elif is_chapter(line):
            chapter = ' '.join(tokens[2:])
            sect, subsect, subsubsect, art, art_code = '', '', '', '', ''
        elif is_section(line):
            sect = ' '.join(tokens[2:])
            subsect, subsubsect, art, art_code = '', '', '', ''
        elif is_subsection(line):
            subsect = ' '.join(tokens[2:])
            subsubsect, art, art_code = '', '', ''
        elif is_subsubsection(line):
            subsubsect = line[1:-1]
            art, art_code = '', ''
        elif is_article(line):
            code = re.search(r'^Article ([\d-]+)', line)[1]
            art, art_code = line, code

    if art != '':
        articles.append((part, chapter, sect, subsect, subsubsect, art, art_code))

    return articles


# In[ ]:


def remove_specials(text: str) -> str:
    special_characters = ['/', '\"', '?', '(', ')', '\'', '-']
    return ''.join([x if (x not in special_characters) else ' ' for x in text])


# In[ ]:
import pandas as pd

def parse_coliee_civil_code(filename: str, remove_punct: bool=True) -> pd.DataFrame:
    with open(filename) as file:
        lines = file.readlines()
        articles = parse_lines(lines[1:])  # Skip the first line (Title)
        articles_df = pd.DataFrame(
            articles,
            columns=['part', 'chap', 'sect', 'subsect', 'subsubsect', 'text', 'docno']
        )

        if remove_punct:
            articles_df['text'] = articles_df['text'].map(remove_specials)

        return articles_df[['text', 'docno', 'part', 'chap',
                            'sect', 'subsect', 'subsubsect']]

def parse_coliee_civil_code_for_encoder(filename: str) -> pd.DataFrame:
    _df = parse_coliee_civil_code(filename, remove_punct=False)
    return _df[['text', 'docno']]


# In[ ]:


articles = parse_coliee_civil_code(ARTICLES_INPUT_DIR_PATH)
encoder_articles = parse_coliee_civil_code_for_encoder(ARTICLES_INPUT_DIR_PATH)


# ## Data Cleaning

# In[ ]:


articles.drop_duplicates(inplace=True)
encoder_articles.drop_duplicates(inplace=True)


# In[ ]:


assert articles['docno'].isna().sum() == 0
assert articles.duplicated().sum() == 0
assert articles.duplicated(subset='docno').sum() == 0

assert encoder_articles['docno'].isna().sum() == 0
assert encoder_articles.duplicated().sum() == 0
assert encoder_articles.duplicated(subset='docno').sum() == 0


# ### Deleted Articles

# In[ ]:


_articles = articles.copy()
_articles.loc[:, 'text_length'] = _articles['text'].map(lambda x: len(x))
display(_articles.sort_values('text_length'))


# In[ ]:


display(articles.loc[articles['text'].str.contains('Deleted'), 'text'])


# #### Remove Deleted Articles

# In[ ]:


deleted_articles_index = articles['text'].str.contains('Deleted')
deleted_encoder_articles_index = encoder_articles['text'].str.contains('Deleted')

articles = articles.loc[~deleted_articles_index]
encoder_articles = encoder_articles.loc[~deleted_encoder_articles_index]


# ## Export... finally

# In[ ]:


articles.to_csv(ARTICLES_OUTPUT_PATH, index=False)
encoder_articles.to_csv(ARTICLES_OUTPUT_ENCODER_PATH, index=False)


# # Parse Training Data

# In[ ]:

from xml.dom.minidom import  NodeList

def get_text(nodelist: NodeList) -> str:
    return ''.join([node.firstChild.nodeValue for node in nodelist])

import os
from glob import glob
from xml.dom.minidom import parse

def parse_coliee_data(path: str, remove_punc: bool=True) -> pd.DataFrame:
    files = glob(os.path.join(path, 'riteval*.xml'))
    qids, labels, queries, art_codes = [], [], [], []

    for file in files:
        with open(file) as f:
            document = parse(f)
            for tag in document.getElementsByTagName('pair'):
                id = tag.getAttribute('id')
                label = tag.getAttribute('label')
                arts = get_text(tag.getElementsByTagName('t1')).strip()
                query = get_text(tag.getElementsByTagName('t2')).strip()
                
                if remove_punc:
                    query = remove_specials(query)

                re_compiler = re.compile(r'^Article ([\d-]+)', re.MULTILINE)
                for code in re_compiler.finditer(arts):
                    qids.append(id)
                    labels.append(label)
                    art_codes.append(code[1])
                    queries.append(query)

    return pd.DataFrame({'qid': qids,
                         'query': queries,
                         'entail': labels,
                         'art_code': art_codes,
                         'label': 1})

def parse_coliee_data_for_encoder(path: str) -> pd.DataFrame:
    _df = parse_coliee_data(path=path, remove_punc=False)
    return _df.loc[:, ['qid', 'query']].drop_duplicates().reset_index()


# In[ ]:


train_data = parse_coliee_data(path=TRAIN_QUERIES_DIR)
test_data = parse_coliee_data(path=TEST_QUERIES_DIR)

encoder_train_query = parse_coliee_data_for_encoder(path=TRAIN_QUERIES_DIR)
encoder_test_query = parse_coliee_data_for_encoder(path=TEST_QUERIES_DIR)
encoder_queries = pd.concat([encoder_train_query, encoder_test_query])


# In[ ]:


train_data.to_csv(TRAIN_QUERIES_OUPTUT_PATH, index=False)
test_data.to_csv(TEST_QUERIES_OUPTUT_PATH, index=False)

encoder_queries.to_csv(QUERIES_OUTPUT_ENCODER_PATH, index=False)

