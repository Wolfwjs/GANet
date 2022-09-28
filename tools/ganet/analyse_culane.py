key="Total Normal Crowd hlight Shadow Noline Arrow Curve Cross Night FPS".lower().split()
s="78.79 93.24 77.16 71.24 77.88 53.59 89.62 75.92 1240 72.75 153".split()
"""78.77 93.10 77.86 70.37 79.26 52.85 89.94 76.58 0.00 73.06 """ # 60 epcoh
"""78.46 93.15 77.03 71.37 80.33 52.75 89.88 77.36 0.00 72.87 """ # 80 epoch

m="79.39 93.73 77.92 71.64 79.49 52.63 90.37 76.32 1368 73.67 127".split()
x="78.98 93.27 77.84 71.60 77.46 52.97 90.15 77.54 1457 73.26" # 60 epoch

l="79.63 93.67 78.66 71.82 78.32 53.38 89.86 77.37 1352 73.85 63".split() # bs=32
'''75.67 90.26 74.72 64.93 73.95 46.75 85.02 70.47 0.00 70.27''' # 40 epoch,bs=8
from glob import glob
import mmcv
import re
# d = dict()
# for k in key:
#     d[k.lower()] = m[]
nm_pa = re.compile(".+_(.*).txt")
f1_pa = re.compile("Fmeasure: (.*)")
fp_pa = re.compile("fp: (\d*)")
items = sorted(glob("tools/ganet/culane_out/*.txt"))
pred = dict()
for item in items:
    mode = re.findall(nm_pa, item)[0]
    txt = mmcv.list_from_file(item)
    f1 = re.findall(f1_pa,"".join(txt))[0]
    pred[mode] = float(f1)*100
print(pred)
for k in key:
    if k=='fps':
        continue
    print(f"{pred[k]:.2f} ",end='')