key = "F1 Accuracy FP FN FPS".lower().split()
s="97.71 95.95 1.97 2.62 153".lower().split()
"""97.55 95.72 1.85 3.04"""

m="97.68 95.87 1.99 2.64 127".lower().split()
"""97.50 95.77 1.82 3.17"""

l="97.45 96.44 2.63 2.47 63".lower().split()
"""97.27 96.42 3.24 2.21"""

# pr_l_250=
from glob import glob
import mmcv
import re,json
# d = dict()
# for k in key:
#     d[k.lower()] = m[]
f1 = re.compile('"F1","value":([\.\d]+)')
fp = re.compile('"FP","value":([\.\d]+)')
fn = re.compile('"FN","value":([\.\d]+)')
acc = re.compile('"Accuracy","value":([\.\d]+)')

pred = dict()
# pr = """[{"name":"Accuracy","value":0.9641931763719084,"order":"desc"},{"name":"FP","value":0.03238677210639849,"order":"asc"},{"name":"FN","value":0.0220764438054158,"order":"asc"},{"name":"F1","value":0.9727410723679646,"order":"asc"}]""" # l,250
# pr = """[{"name":"Accuracy","value":0.9577074988873955,"order":"desc"},{"name":"FP","value":0.0182242990654206,"order":"asc"},{"name":"FN","value":0.03166187395159358,"order":"asc"},{"name":"F1","value":0.975010616599403,"order":"asc"}]""" # m,250
pr = """[{"name":"Accuracy","value":0.9572062964294186,"order":"desc"},{"name":"FP","value":0.018475916606757768,"order":"asc"},{"name":"FN","value":0.030433740714114546,"order":"asc"},{"name":"F1","value":0.9755085278397078,"order":"asc"}]""" # s,250
f1= re.findall(f1,pr)[0]
fp= re.findall(fp,pr)[0]
fn= re.findall(fn,pr)[0]
acc= re.findall(acc,pr)[0]
for i in [f1,acc,fp,fn]:
    i = float(i)*100
    print(f"{i:.2f} ",end='')
