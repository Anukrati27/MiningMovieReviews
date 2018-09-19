'''
Created on Aug 23, 2018

@author: anukr
'''
import socket
import GlobalVars
import TrainSet




#Server class listens for client requests and create a separate thread to handle each client
class Server:

    def __init__(self):
        self.host = socket.gethostname()
        self.port = 51800
        self.mySocket = socket.socket()
        self.mySocket.bind((self.host,self.port))
        #self.mySocket.settimeout(0.1)
        print ("Please connect to server at: ",self.mySocket.getsockname())
        self.trainSet = TrainSet.TrainSet()
        self.trainSet.processTrainingData()
    

    def startServer(self):
        try:
            flag = True
            while flag:
                try: 
                    self.mySocket.listen(1)
                except KeyboardInterrupt:
                    self.mySocket.close()
                    flag = False
                    exit(0)
                conn, addr = self.mySocket.accept()
                #conn.settimeout(None)
                print ("Connection from: " + str(addr))
                while(1):
                    data = conn.recv(1024).decode()
                    print(data)
                    if data == "quit":
                        conn.close()
                        break
                    
                    if data.startswith("k-mean "):
                        try:
                            k = int(data.strip().split(" ")[1])
                            self.trainSet.kMeanAlgorithm(k)
                            ret = "Done"
                        except Exception as e:
                            ret = "Error--> " + str(e)
                    elif data.startswith("knn "):
                        try:
                            k = int(data.strip().split(" ")[1])
                            self.trainSet.kNNAlgorithm(k)
                            ret = "Done"
                        except Exception as e:
                            ret = "Error--> " + str(e)
                    elif data.startswith("testSet"):
                        ret = self.trainSet.processTestSet()
                        
                    elif data.startswith("recommendationSet"):
                        ret = self.trainSet.processRecommendationSet()
                    else:
                        ret = "Invalid data"
    
                    conn.send(ret.encode())
        except KeyboardInterrupt:
            print("\nServer Closed Successfully.\n")
            exit(0)
        except Exception as e:
            print ("Error in Running Server " + str(e))
            exit(0)

    def __del__(self):
        self.mySocket.close()