try: 
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
   
pred_range = [2, 42]

dongbei_lon_range=[0., 1000.]  #
dongbei_lat_range=[40.4, 1000.] # beijing to ~
#huabei_lon_range=[112.4, 118.3]  #taiyuan-changsha to tangshan
huabei_lon_range=[112.4, 1000.]  #taiyuan-changsha to sea
huabei_lat_range=[34.15, 40.4] # xi'an-xuzhou to beijing

#xibei_lon_range=[103.6, 112.4]  #lanzhou-chengdu to taiyuan-changsha
xibei_lon_range=[0., 112.4]  #mountain to taiyuan-changsha
xibei_lat_range=[34.15, 40.4] # xi'an-xuzhou to beijing

huadong_lon_range=[112.4, 1000.] # taiyuan-changsha to sea
#huadong_lat_range=[28.1, 34.15] # changsha to xi'an-xuzhou
huadong_lat_range=[29.3, 34.15] # chongqing to xi'an-xuzhou

#huaxi_lon_range=[103.6, 112.4] # lanzhou-chengdu to taiyuan-changsha
huaxi_lon_range=[0., 112.4] # mountain to taiyuan-changsha
huaxi_lat_range=[29.3, 34.15] # chongqing to xi'an-xuzhou

huanan_lon_range=[0., 1000.] # mountain to sea
huanan_lat_range=[0., 29.3] # sea to chongqing

huabeihuadong_lon_range=[112.4, 1000.] # taiyuan-changsha to sea
huabeihuadong_lat_range=[28.1, 40.4] # changsha to beijing

area2lonlat = OrderedDict([
                     ('dongbei', (dongbei_lon_range, dongbei_lat_range)),
                     ('huabei', (huabei_lon_range, huabei_lat_range)),
                     ('xibei', (xibei_lon_range, xibei_lat_range)),
                     ('huadong', (huadong_lon_range, huadong_lat_range)),
                     ('huaxi', (huaxi_lon_range, huaxi_lat_range)),
                     ('huanan', (huanan_lon_range, huanan_lat_range)),
                     ])

haerbin_stations = [str(i)+'A' for i in range(1129, 1141)]  #12
changchun_stations = [str(i)+'A' for i in range(1119, 1129)]  #10
shenyang_stations = [str(i)+'A' for i in range(1098, 1109)]  #11

beijing_stations = [str(i)+'A' for i in range(1001, 1013)]  #12
#langfang_stations = [str(i)+'A' for i in range(1067, 1071)]  #4
tianjin_stations = [str(i)+'A' for i in range(1013, 1028)]  #15
tangshan_stations = [str(i)+'A' for i in range(1036, 1042)]  #6

baoding_stations = [str(i)+'A' for i in range(1051, 1057)]  #6
shijiazhuang_stations = [str(i)+'A' for i in range(1028, 1036)]  #8
#hengshui_stations = [str(i)+'A' for i in range(1074, 1077)]  #3

xingtai_stations = [str(i)+'A' for i in range(1077, 1081)]  #4
handan_stations = [str(i)+'A' for i in range(1047, 1051)]  #4

jinan_stations = [str(i)+'A' for i in range(1299, 1307)]  #8

xian_stations = [str(i)+'A' for i in range(1462, 1475)] #13

chongqing_stations = [str(i)+'A' for i in range(1414, 1431)] #17
chengdu_stations = [str(i)+'A' for i in range(1431, 1439)] #18

nanjing_stations = [str(i)+'A' for i in range(1151, 1160)] #9
shanghai_stations = [str(i)+'A' for i in range(1141, 1151)] #10
hangzhou_stations = [str(i)+'A' for i in range(1223, 1234)] #11
hefei_stations = [str(i)+'A' for i in range(1270, 1280)] #10
wuhan_stations = [str(i)+'A' for i in range(1325, 1335)] #10
nanchang_stations = [str(i)+'A' for i in range(1290, 1299)] #9
changsha_stations = [str(i)+'A' for i in range(1335, 1345)] #10
guangzhou_stations = [str(i)+'A' for i in range(1345, 1356)] #11

city2stations = OrderedDict([
                   ('haerbin', haerbin_stations),
                   ('changchun', changchun_stations),
                   ('shenyang', shenyang_stations),
                   
                   ('beijing', beijing_stations),
                   ('tianjin', tianjin_stations),
                   ('tangshan', tangshan_stations),
                   ('baoding', baoding_stations),
                   ('shijiazhuang', shijiazhuang_stations),
                   ('xingtai+handan', xingtai_stations + handan_stations),
                   ('jinan', jinan_stations),
                   
                   ('xian', xian_stations),
                   
                   ('nanjing', nanjing_stations),
                   ('shanghai', shanghai_stations),
                   ('hangzhou', hangzhou_stations),
                   ('hefei', hefei_stations),
                   ('wuhan', wuhan_stations),
                   
                   ('chongqing', chongqing_stations),
                   ('chengdu', chengdu_stations),

                   ('nanchang', nanchang_stations),
                   ('changsha', changsha_stations),
                   ('guangzhou', guangzhou_stations),
                   ])

dongbei_cities = ['haerbin', 'changchun', 'shenyang']
huabei_cities = ['beijing', 'tianjin', 'tangshan', 'baoding', 'shijiazhuang', 'xingtai+handan', 'jinan']
xibei_cities = ['xian',]
huadong_cities = ['nanjing', 'shanghai', 'hangzhou', 'hefei', 'wuhan']
huaxi_cities = ['chongqing', 'chengdu']
huanan_cities = ['nanchang', 'changsha', 'guangzhou']
