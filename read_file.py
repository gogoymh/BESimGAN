import csv
import numpy as np
import matplotlib.pyplot as plt

g_loss = np.empty((1500,10))
d_loss = np.empty((1500,10))
time = np.empty((1500,1))

name_list = list(range(10,15010,10))

path = "C:/2019first/고려대/컴퓨터비전/exp57(BEGAN 15000)/%d_loss.txt"

for idx, name in enumerate(name_list):
    f = open(path % name,
             'r', encoding='utf-8', newline='')

    lines = csv.reader(f, delimiter='\t')

    a = [line for line in lines]
    g_tmp = a[1]
    d_tmp = a[3]
    time_tmp = float(a[5][0])

    for i in range(10):
        g_tmp[i] = float(g_tmp[i])
        d_tmp[i] = float(d_tmp[i])
    
    g_loss[idx] = np.array(g_tmp)
    d_loss[idx] = np.array(d_tmp)
    time[idx] = np.array(time_tmp)

g_loss = np.reshape(g_loss, (15000))
d_loss = np.reshape(d_loss, (15000))
time = np.reshape(time, (1500))

plt.plot(g_loss)
plt.plot(d_loss)
plt.axis([0,15000,0,max(g_loss.max(),d_loss.max())])
plt.savefig("C:/2019first/고려대/컴퓨터비전/BEGAN15000.png")
plt.show()
plt.close()











