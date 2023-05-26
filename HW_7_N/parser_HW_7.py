import re

import requests

from bs4 import BeautifulSoup


for i in range(1, 100):
    site = f'your_link{i}'
    response = requests.get(site)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    image_tags = soup.find_all('img')
    urls = [img['src'] for img in image_tags]
    for url in urls:
        filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
        if not filename:
            print("Regular expression didn't match with the url: {}".format(url))
            continue
        with open(filename.group(1), 'wb') as f:
            if 'http' not in url:
                url = '{}{}'.format(site, url)
            response = requests.get(url)
            f.write(response.content)
    print("Download complete, downloaded images can be found in current directory!")
