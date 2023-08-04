import requests 
from bs4 import BeautifulSoup
url = 'https://vnexpress.net'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
article_links = soup.article.h3.a
article_url = article_links['href']
print(article_url)
article_response = requests.get(article_url)
article_soup = BeautifulSoup(article_response.content, 'html.parser')
article_title = article_soup.find('h1').contents
article_description = article_soup.find('h2').contents
article_detail = article_soup.find_all('p')
article_image = article_soup.find_all('img')
print('Title:', article_title)
print('Description:', article_description)
for detail in article_detail:
    print('Detail:', detail.get_text())
for image in article_image:
    print('Image:', image.get('src'))
