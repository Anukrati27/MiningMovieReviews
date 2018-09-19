'''
Created on Aug 21, 2018

@author: anukr
'''
import socket

class Client:

    def __init__(self, host, port):
        self.host = socket.gethostbyname(host)
        self.port = port
        self.mySocket = None

    def startClient(self):
        try:
            self.mySocket = socket.socket()
            self.mySocket.connect((self.host, self.port))
            #self.mySocket.settimeout(10)
        except Exception as e:
            print ("Error in connection to server " + str(e))
            print ("Please check the ip-address or Port Number!!")
            exit(0)

    def sendMessage(self, message):
        try:
            if "quit" == message.lower():
                self.mySocket.send(message.encode())
                self.stopClient()
            else:
                self.mySocket.send(message.encode())
                data = self.mySocket.recv(1024).decode()
                print("Data Received--> ", data)
        except Exception as e:
            print ("Error in parsing message " + str(e))
            exit(0)

    def stopClient(self):
        self.mySocket.close()