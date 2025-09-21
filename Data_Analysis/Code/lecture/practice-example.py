# %%
# install the important packages
# pip install requests 
# pip install lxml
# pip install pandas
# pip install beautifulsoup4

# check what packages are installed
# pip list

# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd

# %%
# basic web scraping for data
url = "https://quotes.toscrape.com/"  # pratice wensite for web scraping
response = requests.get(url) # use request to get the html text
print("1. response code: ", response.status_code)   # check if the response is 200
response_txt = response.text
print("2. first 500 words: \n", response_txt[:500])  # the 500 words of the html

# %%
# turn txt to beautifulsoup object -> lxml DOM tree
                                        # benifit: can use selector to find the element
soup = BeautifulSoup(response_txt, "lxml")  
# fetch all the quote blocks
quotes = soup.find_all("div", class_="quote")
# See the first quote block
print(quotes[0])
# see the total number of quotes
print("find how many lenth:", len(quotes))

# %%
# testing to fetch the first quote
q = quotes[0]  # first quote block
text = q.find("span", class_="text").get_text()
author = q.find("small", class_="author").get_text()
tags = [tag.get_text() for tag in q.find_all("a", class_="tag")]
print("context:", text)
print("author:", author)
print("tag:", tags)
print("-"*30)

# %%
for q in quotes[:3]: # only the first 3 quotes
    text = q.find("span", class_="text").get_text()
    author = q.find("small", class_="author").get_text()
    tags = [tag.get_text() for tag in q.find_all("a", class_="tag")]
    print("sentence:", text)
    print("author:", author)
    print("tag:", tags)
    print("-"*30)

# %%
# wrap the above code into a function to reuse!
def parse_quotes(html):
    data = []
    for q in soup.select("div.quote"):
        # use slect_one to get the tag that you are looking for
        text = q.select_one("span.text").get_text(strip=True)
        author = q.select_one("small.author").get_text(strip=True)
        
        # store the data in a list of dictionary
        data.append({"text": text, "author": author})
    return data

# testing function
quotes = parse_quotes(response_txt)

# check if the funcion works
print("-"*30)
print(len(quotes))  # total number of quotes
quotes[:3]  # first 3 quotes
print("-"*30)

# %%
# save the data into a file(csv)
import csv

with open("quotes.csv", "w", newline="", encoding="utf-8") as f:
    # writer: create a csv writer object to read and write the file
    writer = csv.writer(f)
    writer.writerow(["text", "author", "tags"])
    
    for q in quotes:
        text = q.find("span", class_="text").get_text()
        author = q.find("small", class_="author").get_text()
        tags = ",".join([tag.get_text() for tag in q.find_all("a", class_="tag")])
        writer.writerow([text, author, tags])

print("save to quotes.csv!")

# %%
# save the data into a file(excel-xlsx)
# pip install openpyxl
import pandas as pd

data = []
for q in quotes:
    text = q.find("span", class_="text").get_text()
    author = q.find("small", class_="author").get_text()
    tags = ",".join([tag.get_text() for tag in q.find_all("a", class_="tag")])
    data.append({"text": text, "author": author, "tags": tags})

df = pd.DataFrame(data)

# save to  Excel
df.to_excel("quotes.xlsx", index=False)

print("save to quotes.xlsx")

# %%
# NOTE: 
## html = requests.get(url).text 
## soupt = BeautifulSoup(html, "lxml")
## soup.select("the tag you want to find")
## soup.select_one("the tag you want to find").get_text(strip=True)