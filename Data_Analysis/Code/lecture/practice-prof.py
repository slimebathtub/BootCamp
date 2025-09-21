# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd

# %%
# test if the web srcapping is working
url = "https://www.ratemyprofessors.com/search/professors/1886?q=*" # pratice wensite for web scraping
response = requests.get(url) # use request to get the html text
print("1. response code: ", response.status_code)   # check if the response is 200
response_txt = response.text
print("2. first 500 words: \n", response_txt[:500])  # the 500 words of the html