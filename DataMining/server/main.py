'''
Created on Aug 9, 2018

@author: anukrati
'''
import Server
import sys
import signal
import sys



def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
    
    
if __name__== "__main__":
    try:
        #checking if required parameters are present
        signal.signal(signal.SIGINT, signal_handler)
        if len(sys.argv) != 1:
            print ("Provide input in following form")
            print ("python Main.py")
            exit(0)
        server = Server.Server()
        server.startServer()
    except Exception as e:
        print("Error while running server: ",str(e))
        exit(0)
