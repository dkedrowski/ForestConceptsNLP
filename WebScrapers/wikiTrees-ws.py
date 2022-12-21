import io
import wikipedia
import time
import pandas as pd
from collections import Counter

wikiTrees_file = io.open('WikipediaTreesG-S.txt', mode='r', encoding='utf-8')
wikiTrees_data = wikiTrees_file.read()
wikiTrees_list = wikiTrees_data.split('\n')
numTrees = len(wikiTrees_list)
wikiTrees_file.close()

ignore_list = [ 'See also', 'References', 'Further reading', 'External links', 'Bibliography', 'Gallery',
                'General references', 'General sources', 'Images', 'List of largest trees', 'List of tallest trees',
                'Literature cited', 'Notes', 'Photo gallery', 'References and external links', 'Sources',
                'Works cited' ]

n = 0
content = []
for tree in wikiTrees_list:
    n += 1
    if len(tree.split()) > 1:
        genus = tree.split()[0]
        species = ' '.join(tree.split()[1:])
    else:
        genus = tree
        species = ''

    try:
        wiki = wikipedia.page(tree)
        text = wiki.content
        sections = wiki.sections

        if wiki.summary != None:
            content.append({'text': wiki.summary,
                            'category': 'Summary',
                            'plant': '',
                            'genus': genus,
                            'species': species,
                            'plant_group': ''})
        for section in sections:
            if section not in ignore_list and wiki.section(section) != None:
                dic = {'text': wiki.section(section),
                       'category': section,
                       'plant': '',
                       'genus': genus,
                       'species': species,
                       'plant_group': ''}
                content.append(dic)
    except wikipedia.exceptions.PageError:
        print(f'Page skipped: {tree}. {n}/{numTrees} attempted')
    except wikipedia.exceptions.HTTPTimeoutError:
        print(f'Timeout error: {tree}. {n}/{numTrees} attempted')
    except wikipedia.exceptions.DisambiguationError:
        print(f'Ambiguous: {tree}. {n}/{numTrees} attempted')
    # time.sleep(1)

wikiTrees_file = io.open('WikipediaTreesPlant.txt', mode='r', encoding='utf-8')
wikiTrees_data = wikiTrees_file.read()
wikiTrees_list = wikiTrees_data.split('\n')
numTrees = len(wikiTrees_list)
wikiTrees_file.close()

n = 0
for tree in wikiTrees_list:
    n += 1

    try:
        wiki = wikipedia.page(tree)
        text = wiki.content
        sections = wiki.sections

        if wiki.summary != None:
            content.append({'text': wiki.summary,
                            'category': 'Summary',
                            'plant': tree,
                            'genus': '',
                            'species': '',
                            'plant_group': ''})
        for section in sections:
            if section not in ignore_list and wiki.section(section) != None:
                dic = {'text': wiki.section(section),
                       'category': section,
                       'plant': tree,
                       'genus': '',
                       'species': '',
                       'plant_group': ''}
                content.append(dic)
    except wikipedia.exceptions.PageError:
        print(f'Page skipped: {tree}. {n}/{numTrees} attempted')
    except wikipedia.exceptions.HTTPTimeoutError:
        print(f'Timeout error: {tree}. {n}/{numTrees} attempted')
    except wikipedia.exceptions.DisambiguationError:
        print(f'Ambiguous: {tree}. {n}/{numTrees} attempted')
    # time.sleep(1)

df = pd.DataFrame.from_dict(content)
df.to_csv('wikipedia_scraped_data.csv', index=False)
print(df.head)
# cat_count = Counter(df['category'])
# print(cat_count)
