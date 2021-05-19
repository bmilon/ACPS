# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 06:38:59 2021

@author: mbhattac
"""


import socket
import os 
from os import listdir
from os.path import isfile, join
from PIL import Image
import random
import numpy as np
import time 
import datetime
def server_program():
    cov = [[1e-7, 0], [0, 1e-7]] 
    BUFFER_SIZE = 4096 
    SEPARATOR = "<SEPARATOR>"
    # get the hostname
    mypath = r'C:\Users\mbhattac\Downloads\results\train\stem_rust'
    onlyfiles = [mypath + '\\' +f for f in listdir(mypath) if isfile(join(mypath, f))]
    s_lat,s_lon=30.93975800, 76.51832152

    host = socket.gethostname()
    port = 5000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    client_socket, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    while True:
        print('------------------------------------------------------------------------------------')
        file_name = random.choice(onlyfiles)
        filesize = os.path.getsize(file_name)
        lat,lon = np.random.multivariate_normal([s_lat,s_lon], cov, 1).T
        lat,lon = lat[0],lon[0]
        moisture_percentage = random.randint(0,100)
        print(lat,lon,str(datetime.datetime.now()),moisture_percentage,filesize,file_name)
        client_socket.send(f"{file_name}{SEPARATOR}{filesize}{SEPARATOR}{lat}{SEPARATOR}{lon}{SEPARATOR}{moisture_percentage}{SEPARATOR}{str(datetime.datetime.now())}".encode())
        
        f = open(file_name, "rb") 
        current_pos = f.tell()
        file_size = os.fstat(f.fileno()).st_size
        while file_size  != f.tell():
            bytes_read = f.read(BUFFER_SIZE)            
            client_socket.sendall(bytes_read)
        time.sleep(10)
    server_socket.close()  # close the connection


if __name__ == '__main__':
    server_program()

    # while True:
    #     # receive data stream. it won't accept data packet greater than 1024 bytes
    #     data = conn.recv(1024).decode()
    #     if not data:
    #         # if data is not received break
    #         break
    #     print("from connected user: " + str(data))
    #     data = input(' -> ')
    #     conn.send(data.encode())  # send data to the client

