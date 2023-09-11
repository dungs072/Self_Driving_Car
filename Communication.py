""" PY READER FROM ESP """
# Written by Junicchi - https://github.com/Kebablord
import socket
from time import sleep
import urllib.request
url = "http://192.168.1.21"  # ESP's IP, ex: http://192.168.102/ (Check serial console while uploading the ESP code, the IP will be printed)

# global data for training
class ConnectionNetwork:
    def __init__(self):
        pass
    def get_data_dist(self):
        """
			format for data = 'forward_dist+left_dist+right_dist'
        """
        data = urllib.request.urlopen(url).read()
        data = data.decode('utf-8')
        f,l,r = [float(x) for x in data.split('+')]
        return (f,l,r)
    
    def send_action(self, action):
        command = url+"/"+action
        urllib.request.urlopen(command)
        
host, port = "192.168.1.8", 25001
class UnRealConnectionNetwork: 
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
    def send_action(self,action:str):
        self.sock.sendall(action.encode("UTF-8")) #Converting string to Byte, and sending it to C#
        receivedData = self.sock.recv(1024).decode("UTF-8") #receiveing data in Byte fron C#, and converting it to String
        #print(receivedData)
    def get_data_dist(self):
        """
			format for data = 'forward_dist+left_dist+right_dist'
        """
        weird = "S"
        self.sock.sendall(weird.encode("UTF-8"))
        data = self.sock.recv(1024).decode("UTF-8")
        f,l,r = [float(x) for x in data.split('+')]
        return (f,l,r)
