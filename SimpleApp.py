#/hw6/kddcup.data 

"""SimpleApp.py"""
from pyspark import SparkContext,SparkConf
import numpy as np
import time

#assuming binary categories for now...
def get_idx(col):
    if(col == 1):
        return 0
    elif(col == 2):
        return 1
    elif(col == 3):
        return 2
    else:
        return 3

def pre_process(line):
    x = line.split(",")
    parsed_line  = []
    word_index = [1,2,3,41]
    #calculated categories with preprocess.py
    categories = [['tcp', 'udp', 'icmp'], ['http', 'smtp', 'domain_u', 'auth', 'finger', 'telnet', 'eco_i', 'ftp', 'ntp_u', 'ecr_i', 'other', 'urp_i', 'private', 'pop_3', 'ftp_data', 'netstat', 'daytime', 'ssh', 'echo', 'time', 'name', 'whois', 'domain', 'mtp', 'gopher', 'remote_job', 'rje', 'ctf', 'supdup', 'link', 'systat', 'discard', 'X11', 'shell', 'login', 'imap4', 'nntp', 'uucp', 'pm_dump', 'IRC', 'Z39_50', 'netbios_dgm', 'ldap', 'sunrpc', 'courier', 'exec', 'bgp', 'csnet_ns', 'http_443', 'klogin', 'printer', 'netbios_ssn', 'pop_2', 'nnsp', 'efs', 'hostnames', 'uucp_path', 'sql_net', 'vmnet', 'iso_tsap', 'netbios_ns', 'kshell', 'urh_i', 'http_2784', 'harvest', 'aol', 'tftp_u', 'http_8001', 'tim_i', 'red_i'], ['SF', 'S2', 'S1', 'S3', 'OTH', 'REJ', 'RSTO', 'S0', 'RSTR', 'RSTOS0', 'SH'], [ 'normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']]
    for idx, i in enumerate(x):
        if(idx in word_index):
            cat_index = get_idx(idx)
            cat_to_numerical = categories[cat_index].index(i)
            parsed_line.append(cat_to_numerical)
        else:
            parsed_line.append(i)
    return np.array([float(x) for x in parsed_line])

#return index of closest center to given point.
def get_closest_pnt(pnt, centers):
    best = 0
    closest = float("+inf")
    for i in range(len(centers)):
        #sum squared dist
        dist = np.sum((pnt - centers[i]) ** 2)
        #if found better one then update
        if closest > dist:
            closest = dist
            best = i
    return best

def get_distance(pnt, c, centers):
    return np.sum((pnt - centers[c]) ** 2)

t0 = time.time()
logFile = "hdfs://cluster01:9000/hw6/1.txt"  #  Should be some file on your hdfs path
conf = SparkConf().setAppName("SimpleApp")
sc = SparkContext(conf = conf)
logData = sc.textFile(logFile).cache()
k = 23
# 
fileuri = "hdfs://cluster01:9000/hw6/kddcup.data_10_percent"
# fileuri = 'hdfs://cluster01:9000/user/a2019403077/kddcup.data.part'
file  = sc.textFile(fileuri)
data = file.map(pre_process).cache()

random_pnts = data.takeSample(False, k, 1)
tmp_dist = 1.0  #random value stolen from random location.
conv_dist = 0.5 #random value stolen from random location.
while tmp_dist > conv_dist:
    #1st get closest center for each point
    closest = data.map( lambda pnt: (get_closest_pnt(pnt, random_pnts), (pnt, 1)) )
    #2nd reduce, output =  23 [center_i, (total pnt, #pnts)], 
    reduce = closest.reduceByKey( lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
    #3th calulate avg pnt of cluster, returns (center, avg)
    avg_pnts = reduce.map( lambda tup: (tup[0], tup[1][0] / tup[1][1]) ).collect()

    #calculate total distance between new and old center..
    tmp_dist = 0
    for t in avg_pnts:
        tmp_dist = tmp_dist + (np.sum((random_pnts[t[0]] - t[1]) ** 2)) #new distance
        random_pnts[t[0]] = t[1] #new random points/centers


# print('Amount of centers: ' + str(len(random_pnts)))
# print("Final centers: " + str(random_pnts))

distance_per_centre = [0] * 23
for idx, pnt in enumerate(data.collect()):
    center = get_closest_pnt(pnt, random_pnts)
    dist = get_distance(pnt, center, random_pnts)
    distance_per_centre[center] = distance_per_centre[center] + dist
   
file = open('results','w') 
file.write("Time elapsed: " +  str(time.time() - t0)) 
file.write('\n')
file.write('Amount of centers: ' + str(len(random_pnts)))
file.write('\n')
file.write("Final centers: " + str(random_pnts)) 
file.write('\n')
for i in distance_per_centre:
    file.write(str(i))
    print('distance: ' + str(i))
    file.write('\n')
 
file.close() 

print("Time elapsed: " +  str(time.time() - t0)) # CPU seconds elapsed (floating point)
