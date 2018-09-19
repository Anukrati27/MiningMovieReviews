from collections import OrderedDict
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import io
import json
import os
import shutil
import re


def intersection(list1, list2):
        if type(list1) == str:
                list1 = [list1]
                element = [element for element in list1 if element in list2]
        else:
                element = [element for element in list1 if element in list2]
        return element


def getgenre(url):
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "lxml")
        main_content = urljoin(url, soup.select(".load-more-data"))
        response = requests.get(main_content)
        broth = BeautifulSoup(response.text, "lxml")
        data = json.loads(soup.find('script', type='application/ld+json').text)
        print(data)
        if 'genre' in data:
                genrenames = data['genre']
        else:
                exit()
        return genrenames
        #check on http://www.imdb.com/title/tt0772154 - no genre issue -->Solved


def getreview(url, genrelist, main_content, i):
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "lxml")
        movie_title = soup.find('title').contents[0]
        movie_title = movie_title.split(" - ")
        movie_title = movie_title[0]
        for char in '–")(:?><.@!#$%^&*/-+}{][|':
                movie_title = movie_title.replace(char, '')
        response = requests.get(main_content)
        broth = BeautifulSoup(response.text, "lxml")
        file = ""
        if genrelist[0] == 'Action':
                a = i
                if os.path.isdir('targetSet\\action\ ' + movie_title) == False:
                        os.makedirs('targetSet\\action\ ' + movie_title)
                fileName = []
                print('trainSet\\action\\relevant')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        fileName.append('trainSet\\action\\relevant\ ' + x + '.txt')
                        file = io.open('trainSet\\action\\relevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file.write("{}\n".format(movie_title))
                        file.write("{}\n".format(review))
                        print("done")
                        file.close()
                        a = a + 1
                """if a < 40:
                        for file in fileName:
                                os.remove(file)"""
                a = i
                fileName = []
                print('trainSet\\animation\\irrelev')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        print(a)
                        if a > 30:
                                break
                        fileName.append('trainSet\\animation\\irrelevant\ ' + x + '.txt')
                        file1 = io.open('trainSet\\animation\irrelevant\ ' + x + '.txt', 'w',
                                       encoding="utf-8")
                        file1.write("{}\n".format(movie_title))
                        file1.write("{}\n".format(review))
                        file1.close()
                        a = a + 1
                a = i
                fileName = []
                print('trainSet\\comedy\\irrelevant')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\\comedy\\irrelevant\ ' + x + '.txt')
                        file2 = io.open('trainSet\comedy\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file2.write("{}\n".format(movie_title))
                        file2.write("{}\n".format(review))
                        file2.close()
                        a = a + 1
                a = i
                fileName = []
                print('trainSet\horror\irrelevant')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\horror\irrelevant\ ' + x + '.txt')
                        file3 = io.open('trainSet\horror\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file3.write("{}\n".format(movie_title))
                        file3.write("{}\n".format(review))
                        a = a + 1
                        file3.close()
                a = i
                print('targetSet\\action\ ')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        file4 = io.open('targetSet\\action\ ' + movie_title + '\\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file4.write("{}\n".format(movie_title))
                        file4.write("{}\n".format(review))
                        file4.close()
                        a = a + 1
                """if a < 40:
                        shutil.rmtree('targetSet\\action\ ' + movie_title)"""

        elif genrelist[0] == 'Animation':
                a = i
                fileName = []
                if os.path.isdir('targetSet\\animation\ ' + movie_title) == False:
                        os.makedirs('targetSet\\animation\ ' + movie_title)
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        fileName.append('trainSet\\animation\\relevant\ ' + x + '.txt')
                        file = io.open('trainSet\\animation\\relevant\ ' + x + '.txt', 'w',
                                       encoding="utf-8")
                        file.write("{}\n".format(movie_title))
                        file.write("{}\n".format(review))
                        file.close()
                        a = a + 1
                """if a < 40:
                        for file in fileName:
                                os.remove(file)"""
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        print(a)
                        if a > 30:
                                break
                        fileName.append('trainSet\\action\\irrelevant\ ' + x + '.txt')
                        file1 = io.open('trainSet\\action\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file1.write("{}\n".format(movie_title))
                        file1.write("Review: {}\n".format(review))
                        file1.close()
                        a = a + 1
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\comedy\irrelevant\ ' + x + '.txt')
                        file2 = io.open('trainSet\comedy\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file2.write("{}\n".format(movie_title))
                        file2.write("{}\n".format(review))
                        file2.close()
                        a = a + 1
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\horror\irrelevant\ ' + x + '.txt')
                        file3 = io.open('trainSet\horror\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file3.write("{}\n".format(movie_title))
                        file3.write("{}\n".format(review))
                        file3.close()
                        a = a + 1
                a = i
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        file4 = io.open('targetSet\\animation\ ' + movie_title + '\\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file4.write("{}\n".format(movie_title))
                        file4.write("{}\n".format(review))
                        file4.close()
                        a = a + 1
                """if a < 40:
                        shutil.rmtree('targetSet\\animation\ ' + movie_title)"""
        elif genrelist[0] == 'Comedy':
                a = i
                if os.path.isdir('targetSet\\comedy\ ' + movie_title) == False:
                        os.makedirs('targetSet\\comedy\ ' + movie_title)
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        fileName.append('trainSet\comedy\\relevant\ ' + x + '.txt')
                        file = io.open('trainSet\comedy\\relevant\ ' + x + '.txt', 'w',
                                       encoding="utf-8")
                        file.write("{}\n".format(movie_title))
                        file.write("{}\n".format(review))
                        file.close()
                        a = a + 1
                """if a < 40:
                        for file in fileName:
                                os.remove(file)"""
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\\action\irrelevant\ ' + x + '.txt')
                        file1 = io.open('trainSet\\action\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file1.write("{}\n".format(movie_title))
                        file1.write("{}\n".format(review))
                        file1.close()
                        a = a + 1
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\\animation\irrelevant\ ' + x + '.txt')
                        file2 = io.open('trainSet\\animation\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file2.write("{}\n".format(movie_title))
                        file2.write("{}\n".format(review))
                        file2.close()
                        a = a + 1
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\horror\irrelevant\ ' + x + '.txt')
                        file3 = io.open('trainSet\horror\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file3.write("{}\n".format(movie_title))
                        file3.write("{}\n".format(review))
                        file3.close()
                        a = a + 1
                a = i
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0]+'_'+str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        file4 = io.open('targetSet\comedy\ ' + movie_title + '\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file4.write("{}\n".format(movie_title))
                        file4.write("{}\n".format(review))
                        file4.close()
                        a = a+1
                """if a < 40:
                        shutil.rmtree('targetSet\comedy\ ' + movie_title)"""
        elif genrelist[0] == 'Horror':
                a = i
                fileName = []
                if os.path.isdir('targetSet\\horror\ ' + movie_title) == False:
                        os.makedirs('targetSet\\horror\ ' + movie_title)
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        fileName.append('trainSet\horror\\relevant\ ' + x + '.txt')
                        file = io.open('trainSet\horror\\relevant\ ' + x + '.txt', 'w',
                                       encoding="utf-8")
                        file.write("{}\n".format(movie_title))
                        file.write("{}\n".format(review))
                        file.close()
                        a = a + 1
                """if a < 40:
                        for file in fileName:
                                os.remove(file)"""
                a = i
                fileName = []
                print('trainSet\\action\irrelevant')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\\action\irrelevant\ ' + x + '.txt')
                        file1 = io.open('trainSet\\action\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file1.write("{}\n".format(movie_title))
                        file1.write("{}\n".format(review))
                        file1.close()
                        a = a + 1
                a = i
                fileName = []
                print('trainSet\comedy\irrelevant')
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\comedy\irrelevant\ ' + x + '.txt')
                        file2 = io.open('trainSet\comedy\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file2.write("{}\n".format(movie_title))
                        file2.write("{}\n".format(review))
                        file2.close()
                        a = a + 1
                a = i
                fileName = []
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a) + '_' + x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        if a > 30:
                                break
                        fileName.append('trainSet\\animation\irrelevant\ ' + x + '.txt')
                        file3 = io.open('trainSet\\animation\irrelevant\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file3.write("{}\n".format(movie_title))
                        file3.write("{}\n".format(review))
                        file3.close()
                        a = a + 1
                a = i
                for item in broth.select(".review-container"):
                        title = item.select(".title")[0].text
                        review = item.select(".text")[0].text
                        x = movie_title.split()
                        x = x[0] + '_' + str(a)+'_'+x[1]
                        print(x)
                        print(review)
                        review = re.sub('[^a-zA-Z0-9 ]',' ', review )
                        file4 = io.open('targetSet\horror\ '+movie_title+ '\ ' + x + '.txt', 'w',
                                        encoding="utf-8")
                        file4.write("{}\n".format(movie_title))
                        file4.write("{}\n".format(review))
                        file4.close()
                        a = a + 1
                """if a < 40:
                        shutil.rmtree('targetSet\horror\ ' + movie_title)"""
        return a


def a(soup,inew,genrelist,i):
        value = i
        data = str(soup.find('div', {'class': 'load-more-data'}))
        if 'data-key' in data:
                if value in range(4):
                        data = soup.find('div', {'class': 'load-more-data'})
                        datakey = data['data-key']
                        main_content = 'https://www.imdb.com//title/'+movie+'/reviews/_ajax?paginationKey='+str(datakey)
                        #print(main_content)
                        res = requests.get(main_content)
                        soup = BeautifulSoup(res.text, "lxml")
                        inew = getreview(url,genrelist,main_content,inew)
                        i = i+1
                        a(soup,inew,genrelist,i)


os.makedirs('trainSet\\action\\relevant')
os.makedirs('trainSet\\animation\\relevant')
os.makedirs('trainSet\comedy\\relevant')
os.makedirs('trainSet\horror\\relevant')
os.makedirs('trainSet\\action\irrelevant')
os.makedirs('trainSet\\animation\irrelevant')
os.makedirs('trainSet\comedy\irrelevant')
os.makedirs('trainSet\horror\irrelevant')

os.makedirs('targetSet\\action')
os.makedirs('targetSet\\animation')
os.makedirs('targetSet\comedy')
os.makedirs('targetSet\horror')
filepath = 'urls_target_test.txt'
with open(filepath) as fp:
        line = fp.readline()
        movieidset = []
        while line:
                movieid = line.split("/")[-2]
                movieidset.append(movieid)
                line = fp.readline()
movieidset = dict.fromkeys(movieidset).keys()

for movie in movieidset:
        f = open("filmlist.txt", "w")
        f.write(movie+'\n')
        url = 'http://www.imdb.com/title/'+movie
        url1 = 'http://www.imdb.com/title/'+movie+'/reviews'
        main_content = url1 + '/_ajax'
        urlreview = url+'/reviews?ref_=tt_urv'
        print(url)
        top_genres = ['Action', 'Animation', 'Comedy', 'Horror']
        genre = getgenre(url)
        print(genre)
        genreintop_genres = intersection(genre, top_genres)
        if len(genreintop_genres) != 0:
                print(genreintop_genres[0])
                i = 0
                temp = 0
                inew = getreview(urlreview,genreintop_genres,main_content, i)
                res = requests.get(main_content)
                soup = BeautifulSoup(res.text, "lxml")
                response = requests.get(main_content)
                broth = BeautifulSoup(response.text, "lxml")
                a(soup, inew, genreintop_genres,temp)
                #getreview(urlreview, genreintop_genres)
        else:
                print("Not a movie in our Selected Genre['Action', 'Animation', 'Comedy', 'Horror']")