# -*- coding:utf-8 -*-
from datetime import datetime, timedelta
import gzip
import cPickle

today = datetime.today()
startdaytime = datetime(2015, 4, 1, 2)
today_string = datetime.today().strftime('%Y%m%d')
startday_string = '20150401'
yesterday_string = (
    datetime.today() -
    timedelta(
        days=1)).strftime('%Y%m%d')



heating_start = datetime(year=today.year, month=3, day=15, hour=2)
heating_end = datetime(year=today.year, month=11, day=15, hour=2)

model_savedir = '/ldata/pm25data/pm25model/rlstm/'
# model_savedir = '/home/xiaoda/projects/keras/examples/pm25/model/'
