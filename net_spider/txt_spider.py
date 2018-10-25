import requests
import sys
from bs4 import BeautifulSoup


class TxtSpider:
    def __init__(self, path: str):
        self._path = path
        self._server_url = 'http://www.biqukan.com'
        self._chapter_list_url = 'http://www.biqukan.com/1_1094/'
        self._chapter_names = list()
        self._chapter_urls = list()
        self._chapter_num = 0

    def __get_chapter_url(self):
        req = requests.get(url=self._chapter_list_url)
        chapter_div_list = BeautifulSoup(req.text, features='html.parser').find_all('div', class_='listmain')
        chapter_a_list = BeautifulSoup(str(chapter_div_list[0]), features='html.parser').find_all('a')
        for element in chapter_a_list[12:]:
            self._chapter_urls.append(self._server_url + element.get('href'))
            self._chapter_names.append(element.string)
        self._chapter_num = len(self._chapter_urls)

    def __get_content(self, url: str):
        req = requests.get(url)
        beautiful_soup = BeautifulSoup(req.text, features='html.parser')
        result = beautiful_soup.find_all('div', class_='showtxt')
        # &nbsp 是html的空格，对应到python是'\xa0'
        return result[0].text.replace('\xa0' * 8, '\n\n')

    def __writer(self, name, text):
        with open(self._path, 'a', encoding='utf-8') as f:
            f.write(name + '\n')
            f.writelines(text)
            f.write('\n\n')

    def download_txt(self):
        self.__get_chapter_url()
        print('一年永恒开始下载')
        for i in range(0, self._chapter_num):
            self.__writer(self._chapter_names[i], self.__get_content(self._chapter_urls[i]))
            sys.stdout.write("  已下载:%.3f%%" % float(i / self._chapter_num) + '\r')
            sys.stdout.flush()
        print('《一年永恒》下载完成')
