import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl
import numpy as np
from datetime import datetime, timedelta

def dst(fecha):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # Fecha en formato yyyymm
    url = 'http://wdc.kugi.kyoto-u.ac.jp/dst_realtime/'+fecha+'/index.html'
    print(url)
    html = urllib.request.urlopen(url, context=ctx).read()
    #Abro el html
    soup = BeautifulSoup(html, 'html.parser')
    #Me fije en que tag esta la data que quiero, se llama pre
    table = soup.find("pre").contents
    #separo por /n (son espacios que hay en determinadas horas), elimino los strings que no me interesan
    u=table[2].split('\n')[7:-1]
    str_list = list(filter(None, u))
    np.savetxt("dst.txt", str_list, delimiter=",", fmt='%s')
    # Separo por espacios
    new_list=list()
    for s in str_list:
        tmp = map(''.join, zip(*[iter(s[3:35])]*4))
        new_list.extend(tmp)
        new_list.append(s[35:40])
        tmp = map(''.join, zip(*[iter(s[40:69])]*4))
        new_list.extend(tmp)
        new_list.append(s[69:74])
        tmp = map(''.join, zip(*[iter(s[74:])]*4))
        new_list.extend(tmp)
        new_list.append(s[-4:])
    dst_data = np.asarray(new_list,dtype=np.float32)

    escala_temporal=[]
    date_start = datetime.strptime(fecha, '%Y%m')
    date_end = date_start + timedelta(days=31)
    while date_start < date_end:
            escala_temporal.append(date_start)
            date_start +=timedelta(hours=1)
    return escala_temporal, dst_data

def kp(file):
    myfile = open(file)
    data_full = []
    for line in myfile:
        data = line.strip('\n')
        data_full.append(data)
    return data_full
