from sklearn.linear_model import LogisticRegression
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mylist = []
mylist += [int(i) for i in input("available resources : ").split()]
print("\n-- maximum resources for each process --")
for i in range(5):
    mylist += [int(j)for j in input("process"+str(i+1)).split()]
print("\n-- allocated resources for each process --")
for i in range(5):
    mylist += [int(j)for j in input("process"+str(i+1)).split()]
mlist = np.load(BASE_DIR +"/trained_data1.npy", allow_pickle=True)
slist= np.load(BASE_DIR + "/trained_data2.npy", allow_pickle=True)
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(mlist,slist)
if(model.predict(np.array([mylist]))[0] == 0):
    print("System is not Safe")
else :
    print("System is Safe")
print("safe=no deadlock,,not safe=deadlock")