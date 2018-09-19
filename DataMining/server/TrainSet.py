'''
Created on Aug 9, 2018

@author: anukrati
'''
import GlobalVars
import re
import glob
from pycorenlp.corenlp import StanfordCoreNLP
import math
from textblob import TextBlob as tb
import os
from idlelib.config import InvalidFgBg
from inspect import ArgSpec
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.manifold import MDS
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable
import webbrowser
import mpld3
from IPython.core.tests.test_formatters import numpy
from builtins import str
from sklearn.neighbors import KNeighborsClassifier
import json


class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        
        

class TrainSet:
    
    def __init__(self):
        host = "http://localhost"
        port = "9000"
        self.k = 4
        self.nlp = StanfordCoreNLP(host + ":" + port)
        self.cachedStopWords = stopwords.words("english")
        self.wordnet_lemmatizer = WordNetLemmatizer()
        #stores all data
        self.reviewData = []
        #self.irrelevantReviewData = []
        #stores aspects from trainset
        self.relevantAspects = set()
        self.relevantExpression = set()
        self.irrelevantAspects = set()
        self.irrelevantExpression = set()
        #stores target aspects
        self.targetRelevantAspects = []
        self.targetRelevantExpression = []
        self.targetIrrelevantAspects = []
        self.targetIrrelevantExpression = []
        #store filteres aspects
        self.represRelevantAspects = []
        self.represRelevantExpression = []
        self.represIrrelevantAspects = []
        self.represIrrelevantExpression = []
        
        self.Tri = set()
        self.Tia = set()
        self.Tre = set()
        self.Tie = set()
        #final calculated data
        self.relevantAspectTfIdf = {}
        self.irrelevantAspectTfIdf = {}
        self.relevantExpressionTfIdf = {}
        self.irrelevantExpressionTfIdf = {}
        
        self.movieNameToAspExpMap = {}
        self.movieNameToGenre = {}
        self.genreMapperToData = {}
        self.testSetMap = {}
        
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        
        self.u = []
        self.v = []
        self.w = []
        self.x = []
        self.movieList = []
        self.TestARecAspectsList = []
        self.TestARecExprList = []
        self.TestARecMovieList = {}
        
        self.test= False
        self.recommend = False
        self.drawMovieList = []
        self.drawMovieListTest = []
        self.drawMovieListRec = []
        self.testSetGenreScore = []
        self.genreScore = []
        self.genreType=[]
        
        
    def readDataFromFile(self, fileName):
        try:
            with open(fileName) as f:
                data = f.readlines()
            
            movieName = data[0]
            review = data[1]
            
            return movieName, review
        except Exception as e:
            pass
    
    def processReviewData(self, review):
        review = review.replace("$", '')
        review = review.replace("#", '')
        url_reg  = r'[a-z]*[:.]+\S+'
        review = re.sub(url_reg, '', review)
        review = review.lower()
        review = ' '.join([word for word in review.split() if word not in self.cachedStopWords])
        #review = ' '.join([self.wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(review) ])
        return review   
    
    def fetchingAspectsAndExpression(self,review):
        output = self.nlp.annotate(
            review,
            properties={
                "outputFormat": "json",
                "annotators": "ner,entitymentions,tokenize,ssplit,pos,lemma"
            }
        )
        asp = []
        exp = []
        for items in output["sentences"]:
            for key, value in items.items():
                if key == "tokens":
                    for item in value:
                        val1 = item["pos"]
                        val2 = item["pos"]
                        if val1 == "nn" or val1 == "NNS" or val1 == "NNP" or val1 == "NNPS" or\
                           val2 == "TITLE" or val2 == "PERSON":
                            asp.append(item["word"])
                        elif val1 == "JJ" or val1 == "JJR" or val1 == "JJS" or\
                             val1 == "RB" or val1 == "RBR" or val1 == "RBS" or\
                             val1 == "VB" or val1 == "VBD" or val1 == "VBG" or \
                             val1 == "VBN" or val1 == "VBP" or val1 == "VBZ" or\
                             val2 == "CRIMINAL_CHARGE":
                            exp.append(item["word"])  
        return asp, exp
    
    def tf(self, word, blob):
        return blob.words.count(word) / len(blob.words)
    
    def n_containing(self, word):
        return sum(1 for blob in self.reviewData if word in blob.words)
    
    def idf(self, word):
        return math.log(len(self.reviewData) / (1 + self.n_containing(word)))
        
            
    def calculateTFIDF(self, word, document):
        tf = self.tf(word, document)
        idf = self.idf(word) 
        return tf*idf

    def computeTFIDF(self):
        for key, value in self.genreMapperToData.items():
            for item in value:
                for k, val in item.items():
                    for word in self.represRelevantAspects[k]:
                        self.relevantAspectTfIdf[key][word] = self.calculateTFIDF(word, self.reviewData[val])
                    for word in self.represIrrelevantAspects[k]:
                        self.irrelevantAspectTfIdf[key][word] = self.calculateTFIDF(word, self.reviewData[val])
                    for word in self.represRelevantExpression[k]:
                        self.relevantExpressionTfIdf[key][word] = self.calculateTFIDF(word, self.reviewData[val])
                        
                    for word in self.represIrrelevantExpression[k]:
                        self.irrelevantExpressionTfIdf[key][word] = self.calculateTFIDF(word, self.reviewData[val])
                        
                        
    def computeFinalAspectsAndExpressionsForRelevantData(self, genre, asp, exp):
        self.represRelevantAspects.append(set(asp) & self.Tra) 
        self.represRelevantExpression.append(set(exp) & self.Tre)
        self.represIrrelevantAspects.append(set(asp) & self.Tia)
        self.represIrrelevantExpression.append(set(exp) & self.Tie)
        self.genreMapperToData[genre].append({len(self.represRelevantAspects)-1 : len(self.reviewData)-1 })
        
    def filterTop100AspectsAndExpression(self):
        for key, value in self.relevantAspectTfIdf.items():
            counter = 0
            tempDict = {}
            while(counter <= 100 and counter < len(value)):
                counter = counter + 1
                maxItem = max(value.items(), key=lambda k: k[1])[0]
                tempDict[maxItem] = value[maxItem]
                value[maxItem] = float("-inf")
            self.relevantAspectTfIdf[key] = tempDict
        
        for key, value in self.irrelevantAspectTfIdf.items():
            counter = 0
            tempDict = {}
            while(counter <= 100 and counter < len(value)):
                counter = counter + 1
                maxItem = max(value.items(), key=lambda k: k[1])[0]
                tempDict[maxItem] = value[maxItem]
                value[maxItem] = float("-inf")
            self.irrelevantAspectTfIdf[key] = tempDict
        for key, value in self.relevantExpressionTfIdf.items():
            counter = 0
            tempDict = {}
            while(counter <= 100 and counter < len(value)):
                counter = counter + 1
                maxItem = max(value.items(), key=lambda k: k[1])[0]
                tempDict[maxItem] = value[maxItem]
                value[maxItem] = float("-inf")
            self.relevantExpressionTfIdf[key] = tempDict
            
        for key, value in self.irrelevantExpressionTfIdf.items():
            counter = 0
            tempDict = {}
            while(counter <= 100 and counter < len(value)):
                counter = counter + 1
                maxItem = max(value.items(), key=lambda k: k[1])[0]
                tempDict[maxItem] = value[maxItem]
                value[maxItem] = float("-inf")
            self.irrelevantExpressionTfIdf[key] = tempDict
                
            
    def processTrainRelevantData(self, path):
        try:
            for file in glob.glob(GlobalVars.trainSetPath + path + "/relevant/*.txt"):
                movieName, review = self.readDataFromFile(file)
                review = self.processReviewData(review)
                asp, exp = self.fetchingAspectsAndExpression(review)
                self.relevantAspects.update(set(asp))
                self.relevantExpression.update(set(exp))
        except Exception as e:
            print("error-->" + str(e))
            
    def processTrainIrrelevantData(self, path):
        try:
            for file in glob.glob(GlobalVars.trainSetPath + path + "/irrelevant/*.txt"):
                movieName, review = self.readDataFromFile(file)
                review = self.processReviewData(review)
                asp, exp = self.fetchingAspectsAndExpression(review)
                self.irrelevantAspects.update(set(asp))
                self.irrelevantExpression.update(set(exp))
        except Exception as e:
            print("error-->" + str(e))
            
    def processTargetRelevantData(self, path, genre):
        #try:
        for file in glob.glob(path + "/*.txt"):
            movieName, review = self.readDataFromFile(file)
            review = self.processReviewData(review)
            self.reviewData.append(tb(review))
            asp, exp = self.fetchingAspectsAndExpression(review)
            if movieName in self.movieNameToAspExpMap.keys():
                self.movieNameToAspExpMap[movieName].append(len(self.represRelevantAspects))
            else:
                self.movieNameToAspExpMap[movieName] = [len(self.represRelevantAspects)]
            self.movieNameToGenre[movieName] = genre
            self.computeFinalAspectsAndExpressionsForRelevantData(genre, asp, exp)
        #except Exception as e:
        #    print("error-->" + str(e))

    def processTargetData(self, path):
        try:
            for x in os.listdir(GlobalVars.targetSetPath + path):
                if os.path.isdir(GlobalVars.targetSetPath + path + "/"+ x):
                    #self.relevantReviewData = []
                    #self.irrelevantReviewData = []
                    self.processTargetRelevantData(GlobalVars.targetSetPath + path + "/" + x , path)
                    #self.processTargetIrrelevantData(GlobalVars.targetSetPath + path + "/" + x + "/irreleavnt")
                    #self.computeTFIDF(path)
        except Exception as e:
            print("error-->" + str(e))
                
    '''def computeGenreScore(self):
        for movie, value in self.movieNameToAspExpMap.items():
            GSra = {}
            GSia = {}
            GSre = {}
            GSie = {}
            relevantAspectSet = set()
            irrelevantAspectSet = set()
            relevantExpSet = set()
            irrelevantExpSet = set()
            for val in value:
                relevantAspectSet.update(self.represRelevantAspects[val])
                irrelevantAspectSet.update(self.represIrrelevantAspects[val])
                relevantExpSet.update(self.represRelevantExpression[val])
                irrelevantExpSet.update(self.represIrrelevantExpression[val])
                
            for genre, aspectMap in self.relevantAspectTfIdf.items():
                GSra[genre] = 0
                for word in relevantAspectSet:
                    if word in aspectMap.keys():
                        GSra[genre] = GSra[genre] + aspectMap[word]

            for genre, aspectMap in self.irrelevantAspectTfIdf.items():
                GSia[genre] = 0
                for word in irrelevantAspectSet:            
                    if word in aspectMap.keys():
                        GSia[genre] = GSia[genre] + aspectMap[word]
                        
            for genre, aspectMap in self.relevantExpressionTfIdf.items():
                GSre[genre] = 0
                for word in relevantExpSet:
                    if word in aspectMap.keys():
                        GSre[genre] = GSre[genre] + aspectMap[word]

            for genre, aspectMap in self.irrelevantExpressionTfIdf.items():
                GSie[genre] = 0
                for word in irrelevantExpSet:
                    if word in aspectMap.keys():
                        GSie[genre] = GSie[genre] + aspectMap[word]

            GSp = {}
            GSe = {}
            GSf = {}
            temp = []
            for key in GSra:
                if (GSra[key] == 0 and GSia[key] == 0):
                    GSp[key] = 0
                else:
                    GSp[key] = (GSra[key] - GSia[key]) #/ (GSra[key] + GSia[key])
                if (GSre[key] == 0 and GSie[key] == 0):
                    GSe[key] = 0
                else:
                    GSe[key] = (GSre[key] - GSie[key]) #/ (GSre[key] + GSie[key])
                GSf[key] = GSp[key] + GSe[key]
                if key == GlobalVars.genre[0]:
                    self.u.append(GSf[key])
                elif key == GlobalVars.genre[1]:
                    self.v.append(GSf[key])
                elif key == GlobalVars.genre[2]:
                    self.w.append(GSf[key])
                else:
                    self.x.append(GSf[key])
            temp.append(GSf[GlobalVars.genre[0]])
            temp.append(GSf[GlobalVars.genre[1]])
            temp.append(GSf[GlobalVars.genre[2]])
            temp.append(GSf[GlobalVars.genre[3]])
            self.genreScore.append(temp)
            self.movieList.append(movie)
            self.genreType.append(self.movieNameToGenre[movie])'''
            
    def kMeanAlgorithm(self, k):
        X = np.array(self.genreScore)
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(X)
        labels = kmeans.predict(X)
        C = kmeans.cluster_centers_
        clusters = kmeans.labels_.tolist()
        
        MDS()
        mds = MDS(n_components=2,  random_state=1)
        
        pos = mds.fit_transform(X)
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        xs, ys = pos[:, 0], pos[:, 1]
        
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#ff0000' , 6: '#800000', 7: '#808000', 8: '#008000',\
                           9: '#00FFFF', 10: '#008080', 11: '#0000FF', 12: '#FF00FF', 13: '#800080'}

        cluster_names = {0: 'Genre1', 
             1: 'Gnere2', 
             2: 'Genre3', 
             3: 'Genre4', 
             4: 'Genre5',
             5: 'Gnere6', 
             6: 'Genre7', 
             7: 'Genre8', 
             8: 'Genre9',
             9: 'Gnere10', 
             10: 'Genre11', 
             11: 'Genre12', 
             12: 'Genre13'
        } 
                 
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=self.movieList)) 
        groups = df.groupby('label')
        
        css = """
            text.mpld3-text, div.mpld3-tooltip {
              font-family:Arial, Helvetica, sans-serif;
            }
            
            g.mpld3-xaxis, g.mpld3-yaxis {
            display: none; }
            
            svg.mpld3-figure {
            margin-left: -200px;}
        """
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.margins(0.05)
        
    
        for name, group in groups:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                             label=cluster_names[name], mec='none', 
                             color=cluster_colors[name])
            ax.set_aspect('auto')
            labels = [i for i in group.title]
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                               voffset=10, hoffset=10, css=css)
            mpld3.plugins.connect(fig, tooltip, TopToolbar())    
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
        ax.legend(numpoints=1)
        
        for i in range(len(df)):
            if df.ix[i]['title'] in self.drawMovieListTest:
                ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=12)  
        
        mpld3.save_html(fig, "plot.html") #show the plot
        webbrowser.open_new_tab('plot.html')
        
        line = "<style>\np {text-align: left;color:#EAE419; font-size: 40px; font-weight:bold; }\n \
                 body {background-color: black}\n</style>\n<p> &emsp;&emsp;&emsp;&emsp; k-mean Clustering</p>"
        with open('plot.html', 'r+') as f:
            file_data = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + file_data)
        
        x = PrettyTable()
        x.field_names = ["Test Movie", "Similar Recommended Movie"]
        x.align["Test Movie"] = "c"
        x.align["Similar Recommended Movie"] = "c"
        x.align = "c"
        x.border = True
        x.header = True
        x.format = True
        x.padding_width = 5
        if self.recommend or self.test:
            for entry in self.testSetGenreScore:
                flag = False
                for i in range(kmeans.n_clusters):
                    clus = X[np.where(kmeans.labels_ == i)]
                    for item in clus:
                        a=[]
                        b=[]
                        for i in entry:
                            a.append(round(i,6))
                        for i in item:
                            b.append(round(i,6))
                        v = numpy.array(a) == numpy.array(b)
                        if v.all():
                            test = self.getItem(entry)
                            str = ""
                            for it in clus:
                                temp = self.getItem(it)
                                if temp != test:
                                    str = str + temp + " "
                            x.add_row([test, str])
                            flag = True
                            break
                    if flag:
                        break
            css = "<head><style>\ntable{width: 100%; border: 2px solid #333; border-spacing: 2px; text-align: center; \
                    background-color: #f1f1c1;color:black}\n table th {border: \     2px solid #333;background-color: green;color:white; \
                    font-family: 'Times New Roman'; font-size: 20px;}\n table tr td {border: 1px solid #333; padding: 4px;}\n p \
                    {text-align: center; background-color: green;color:white; font-size: 30px; }\n body {background-color: black}\n \
                    </style></head>\n<p>K-Mean Clustering</p>"
            with open('k_mean.html', 'w') as f: 
                f.write(css)
                f.write(x.get_html_string()) 
            webbrowser.open_new_tab('k_mean.html')
        
        
    def kNNAlgorithm(self, k):
        X = np.array(self.genreScore)
        Y = np.array(self.genreType)
        classifier = KNeighborsClassifier(n_neighbors=k)  
        classifier.fit(X, Y)
        Z = np.array(self.testSetGenreScore)
        nei = classifier.kneighbors(Z, return_distance=False)
        tab = PrettyTable([],auto_width=True, border=True, header_color='yellow,bold',
                    left_padding_width=5, right_padding_width=5)
        fieldList = ["Test Movies"]
        for i in range(k):
            fieldList.append("k = "+str(i+1))
        
        tab.field_names = fieldList
        print(len(fieldList))
        tab.format = True
        tab.header_color='yellow,bold'
        tab.title = 'Results for KNN Algorithm'
        tab.padding_width = 1
        
        for j in range(len(nei)):
            temp = []
            temp.append(self.getItem(Z[j]))
            for item in nei[j]:
                it = self.getItem(X[item])
                temp.append(it)
            print(len(temp))
            tab.add_row(temp)
        
        tab.get_html_string()
        css = "<head><style>\ntable{width: 100%; border: 2px solid #333; border-spacing: 2px; text-align: center; \
                    background-color: #f1f1c1;color:black}\n table th {border: \     2px solid #333;background-color: green;color:white; \
                    font-family: 'Times New Roman'; font-size: 20px;}\n table tr td {border: 1px solid #333; padding: 4px;}\n p \
                    {text-align: center; background-color: green;color:white; font-size: 30px; }\n body {background-color: black}\n \
                    </style></head>\n<p>K-NN Algorithm</p>"
        with open('knn.html', 'w') as f: 
            f.write(css)
            f.write(tab.get_html_string()) 
        webbrowser.open_new_tab('knn.html')
        

    def getItem(self, item):
        for i in range(len(self.genreScore)):
            a=[]
            b=[]
            for j in item:
                a.append(round(j,6))
            for j in self.genreScore[i]:
                b.append(round(j,6))
            v = numpy.array(a) == numpy.array(b)
            if v.all():
                return self.movieList[i]
        return "Item Not Found"
        
    def resetParameters(self):
        
        self.relevantAspects = set()
        self.relevantExpression = set()
        self.irrelevantAspects = set()
        self.irrelevantExpression = set()
        
        self.Tri = set()
        self.Tia = set()
        self.Tre = set()
        self.Tie = set()
                
    def processTrainingData(self):
        for item in GlobalVars.genre:
            self.relevantAspectTfIdf[item] = {}
            self.irrelevantAspectTfIdf[item] = {}
            self.relevantExpressionTfIdf[item] = {}
            self.irrelevantExpressionTfIdf[item] = {}
            self.genreMapperToData[item] = []
            self.processTrainRelevantData(item)
            self.processTrainIrrelevantData(item)
            self.Tra = self.relevantAspects - self.irrelevantAspects
            self.Tia = self.irrelevantAspects - self.relevantAspects
            self.Tre = self.relevantExpression - self.irrelevantExpression
            self.Tie = self.irrelevantExpression - self.relevantExpression
            self.processTargetData(item)
            self.resetParameters()

        self.computeTFIDF()
        self.filterTop100AspectsAndExpression()
        #self.computeGenreScore()
        print("Train Set and Target Set Analysed")
        
    def computeGenreScoreForTrainAndRecommendationData(self, flag=2):
        for movie, value in self.TestARecMovieList.items():
            GSra = {}
            GSia = {}
            GSre = {}
            GSie = {}
            aspectSet = set()
            expSet = set()
            for val in value:
                aspectSet.update(self.TestARecAspectsList[val])
                expSet.update(self.TestARecExprList[val])

            for genre, aspectMap in self.relevantAspectTfIdf.items():
                GSra[genre] = 0
                for word in aspectSet:
                    if word in aspectMap.keys():
                        GSra[genre] = GSra[genre] + aspectMap[word]
                    
            
            for genre, aspectMap in self.irrelevantAspectTfIdf.items():
                GSia[genre] = 0
                for word in aspectSet:            
                    if word in aspectMap.keys():
                        GSia[genre] = GSia[genre] + aspectMap[word]
                        
            for genre, aspectMap in self.relevantExpressionTfIdf.items():
                GSre[genre] = 0
                for word in expSet:
                    if word in aspectMap.keys():
                        GSre[genre] = GSre[genre] + aspectMap[word]

            for genre, aspectMap in self.irrelevantExpressionTfIdf.items():
                GSie[genre] = 0
                for word in expSet:
                    if word in aspectMap.keys():
                        GSie[genre] = GSie[genre] + aspectMap[word]

            GSp = {}
            GSe = {}
            GSf = {}
            temp = []
            for key in GSra:
                if (GSra[key] == 0 and GSia[key] == 0):
                    GSp[key] = 0
                else:
                    #GSp[key] = (GSra[key] - GSia[key]) / (GSra[key] + GSia[key])
                    GSp[key] = (GSra[key] - GSia[key])
                if (GSre[key] == 0 and GSie[key] == 0):
                    GSe[key] = 0
                else:
                    GSe[key] = (GSre[key] - GSie[key]) #/ (GSre[key] + GSie[key])
                GSf[key] = GSp[key] + GSe[key]
                
                if key == GlobalVars.genre[0]:
                    self.u.append(GSf[key])
                elif key == GlobalVars.genre[1]:
                    self.v.append(GSf[key])
                elif key == GlobalVars.genre[2]:
                    self.w.append(GSf[key])
                else:
                    self.x.append(GSf[key])
                    
            temp.append(GSf[GlobalVars.genre[0]])
            temp.append(GSf[GlobalVars.genre[1]])
            temp.append(GSf[GlobalVars.genre[2]])
            temp.append(GSf[GlobalVars.genre[3]])
            self.movieList.append(movie)
            self.testSetMap[movie] = GSf
            self.genreScore.append(temp)
            if flag == 1:
                temp = []
                for key, value in GSf.items():
                    temp.append(value)
                self.testSetGenreScore.append(temp)
                
            self.genreType.append("")
    
    def processTestData(self, path, flag):
        try:
            for file in glob.glob(path + "/*.txt"):
                movieName, review = self.readDataFromFile(file)
                review = self.processReviewData(review)
                asp, exp = self.fetchingAspectsAndExpression(review)
                if flag == 1:
                    self.drawMovieListTest.append(movieName)
                else:
                    self.drawMovieListRec.append(movieName)
                    
                if movieName in self.TestARecMovieList.keys():
                    self.TestARecMovieList[movieName].append(len(self.TestARecAspectsList))
                else:
                    self.TestARecMovieList[movieName] = [len(self.TestARecAspectsList)]
    
                self.TestARecAspectsList.append(asp)
                self.TestARecExprList.append(exp)
        except Exception as e:
            print("error-->" + str(e))
            
    def processTrainAndRecommendationData(self, path, flag):
        try:
            for x in os.listdir(path):
                if os.path.isdir( path + x):
                    self.processTestData(path + x, flag)
        except Exception as e:
            print("error-->" + str(e))
            
    
            
    def processTestSet(self):
        
        #if self.test == True:
        #    self.drawMovieList = self.drawMovieListTest
        #    self.kMeanAlgorithm(self.k)
        #    return "Already processed"
        self.genreScore = []
        self.testSetGenreScore = []
        self.genreType = []
        self.test = True
        self.TestARecAspectsList = []
        self.TestARecExprList = []
        self.TestARecMovieList = {}
        self.processTrainAndRecommendationData(GlobalVars.testSetPath, 1)
        self.computeGenreScoreForTrainAndRecommendationData(1)
        return json.dumps(self.testSetMap)
    
        
    def processRecommendationSet(self):
        #if self.recommend == True:
        #    self.drawMovieList = self.drawMovieListRec
        #    self.kMeanAlgorithm(self.k)
        #    return "Already processed"
        self.recommend = True
        self.TestARecAspectsList = []
        self.TestARecExprList = []
        self.TestARecMovieList = {}
        self.processTrainAndRecommendationData(GlobalVars.recommSetPath, 2)
        self.computeGenreScoreForTrainAndRecommendationData()
        return "done"
    
