# -*- coding:utf-8 -*-

from __future__ import division
import time
import os
import numpy as np
import gzip
import cPickle
from pymongo import MongoClient
client = MongoClient('10.163.174.61')
db = client.wp
pm25 = db.pm25
pm25_station = db.pm25_station
from datetime import datetime,timedelta
import time


start_timestring = '2015040100'
# start_timestring = '2015122500'
# you could specify it again, but i just hardcode this begining
start = time.mktime(time.strptime(start_timestring, '%Y%m%d%H'))
end_time_norm = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
# you could specify it use 2015-08-19 like that
end_timestring = (datetime.strptime(end_time_norm, '%Y-%m-%d')).strftime('%Y%m%d%H')
end = time.mktime(time.strptime(end_timestring, '%Y%m%d%H'))
length = (end-start)/3600

def main():
    pst = []  
    for x in pm25_station.find():
        pst.append(x['Id'])
    
    data = {}
    for ID in pst:
        one = [0 for x in range(int(length))]
	data[ID] = one

    res = pm25.find({"time_point":{"$gt":start,"$lt":end}})
    num = 0
    for x in res:
	ID = x['Id']
	ind = (x['time_point'] - start)/3600
	if ID in pst:
	    data[ID][int(ind)] = x['data']['pm2_5']
	num = num +1 
	if num%1000==0:
	    print num
		

    f = gzip.open('data_source/' + start_timestring + '_' + end_timestring +'pm25.pkl.gz', 'wb')
    f.write(cPickle.dumps(data))
    f.close()
   
    
    

if __name__ == '__main__':
    main()
