from bs4 import BeautifulSoup

import os
import pandas as pd
import re
import requests


base_url  = "http://papers.nips.cc"

index_urls = {1987: "https://papers.nips.cc/book/neural-information-processing-systems-1987"}
for i in range(1,30):
    year = i+1987
    index_urls[year] = "http://papers.nips.cc/book/advances-in-neural-information-processing-systems-%d-%d" % (i, year)

papers = list()


for year in sorted(index_urls.keys()):
    index_url = index_urls[year]
    index_html_path = os.path.join("working", "html", str(year)+".html")

    if not os.path.exists(index_html_path):
        r = requests.get(index_url)
        if not os.path.exists(os.path.dirname(index_html_path)):
            os.makedirs(os.path.dirname(index_html_path))
        with open(index_html_path, "wb") as index_html_file:
            index_html_file.write(r.content)
    with open(index_html_path, "rb") as f:
       html_content = f.read()
    soup = BeautifulSoup(html_content, "lxml")
    paper_links = [link for link in soup.find_all('a') if link["href"][:7]=="/paper/"]
    print("%d Papers Found" % len(paper_links))


    temp_path = os.path.join("working", "temp.txt")

    for link in paper_links:
        paper_title = link.contents[0]
        info_link = base_url + link["href"]
        #pdf_link = info_link + ".pdf"
        pdf_name = link["href"][7:] + ".pdf"
        paper_id = re.findall(r"^(\d+)-", pdf_name)[0]
        #print(year, " ", paper_id) #paper_title.encode('ascii', 'namereplace')

        paper_info_html_path = os.path.join("working", "html", str(year), str(paper_id)+".html")
        if not os.path.exists(paper_info_html_path):
            r = requests.get(info_link)
            if not os.path.exists(os.path.dirname(paper_info_html_path)):
                os.makedirs(os.path.dirname(paper_info_html_path))
            with open(paper_info_html_path, "wb") as f:
                f.write(r.content)
                
        with open(paper_info_html_path, "rb") as f:
           html_content = f.read()
           
        paper_soup = BeautifulSoup(html_content, "lxml")
        try: 
            abstract = paper_soup.find('p', attrs={"class": "abstract"}).contents[0]
        except:
            print("Abstract not found %s" % paper_title.encode("ascii", "replace"))
            abstract = ""


        papers.append([paper_title, abstract, ])


pd.DataFrame(papers, columns=[ "title", "abstract"]).to_csv("output/papers.csv", index=False)

