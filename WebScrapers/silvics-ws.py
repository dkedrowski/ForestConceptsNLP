from bs4 import Tag, NavigableString, BeautifulSoup
import requests
import pandas as pd

# Create volume specific tuples to pass in a list to a for loop
url1 = 'https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_1/vol1_table_of_contents.htm'
plant_group1 = 'conifer'
url_path1 = 'https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_1/'
tup1 = (url1, plant_group1, url_path1)

url2 = 'https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_2/vol2_table_of_contents.htm'
plant_group2 = 'hardwood'
url_path2 = 'https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_2/'
tup2 = (url2, plant_group2, url_path2)

tup_list = [ tup1, tup2 ]

lst_labels = []
for tup in tup_list:
    url = tup[0]
    plant_group = tup[1]
    url_path = tup[2]

    req = requests.get(url)
    content = req.text
    soup = BeautifulSoup(content, 'html.parser')

    # Get links from contents page
    links_temp = []
    for link in soup.find_all('a'):
        links_temp.append(url_path + link.get('href'))

    # Only keep links that occur more than once on the contents page
    links = []
    for link in links_temp:
        if links_temp.count(link) > 1:
            links.append(link)
    links = list(set(links)) # removes duplicates

    # loop through the links to extract the information
    for link in links:
        print(link)
        req = requests.get(link)
        content = req.text
        soup = BeautifulSoup(content, 'html.parser')

        # extract genus and species from the url link
        genus_species = link.replace(url_path, '').replace('.htm', '').replace('/', ' ')
        genus = genus_species.split()[0]
        species = genus_species.split()[1]

        # attempt to extract the common name from the html content
        if content.find('<FONT SIZE="+4">') > 0:
            plant_start = content.find('<FONT SIZE="+4">') + len('<FONT SIZE="+4">')
            plant_end = content.find('</FONT>', plant_start)
            plant = content[plant_start:plant_end]
            plant = plant.replace('<B>', '').replace('</B>', '').strip()
        else:
            plant_start = content.find('<FONT SIZE="+3">') + len('<FONT SIZE="+3">')
            plant_end = content.find('</FONT>', plant_start)
            plant = content[plant_start:plant_end]
            plant = plant.replace('<B>', '').replace('</B>', '').strip()
            plant = plant.replace('<I>', '').replace('</I>', '').strip()

        # put all h2 and h3 entries in a list
        labels = soup.select('h3')
        labels.extend(soup.select('h2'))
        # labels = soup.select('body')
        # print(labels)

        # extract text based on h2 and h3 entries
        for tag in labels:
            lbl = []
            for x in tag.next_siblings:
                if x.name == 'h3' or x.name == 'h2':
                    break
                elif x.name == 'p':
                    lbl.append(x.get_text())

            # convert list to string, replace end-of-line characters, clean up string
            text_content = ' '.join(lbl)
            text_content = text_content.replace('\n', '')
            text_content = text_content.replace('\r', '')
            text_content = ' '.join(text_content.split())

            # format information as a dictionary and append to a list
            if text_content != "":
                dic = {'text': text_content, 'category':tag.get_text(), 'plant':plant, 'genus':genus, 'species':species, 'plant_group':plant_group}
                lst_labels.append(dic)

# convert the list of information to a dataframe and output as a csv file
df = pd.DataFrame.from_dict(lst_labels)
# print(df)
df.to_csv('silvics_scraped_data.csv', index=False)