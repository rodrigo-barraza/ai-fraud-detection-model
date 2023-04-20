import requests
import time
import socket

def for_topics(topics_list, host='localhost', port='29080'):
    ready = False

    while ready == False:
        
        try:
            r = requests.get(url='http://{}:{}/topics'.format(host, port))
            current_topics = list(r.json())
            if all(topic in current_topics for topic in topics_list):
                ready = True
                print('Kafka Is Ready')
        except:
            time.sleep(1)
            
            
def for_host(port, host='localhost'):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    ready = False
    
    while ready == False:
        try:
            s.connect((host, port))
            s.shutdown(2)
            print("Success connecting to ")
            print(host," on port: ",str(port))
            ready = True
        except socket.error as e:
            print("Cannot connect to ")
            print(host," on port: ",str(port))
            print(e)
            time.sleep(1)