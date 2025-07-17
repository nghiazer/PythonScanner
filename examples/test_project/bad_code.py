# Bad example with convention violations
import os,sys
from pathlib import *

def calcAvg(nums):
    return sum(nums)/len(nums)

class dataProcessor:
    def __init__(self):
        self.Data=[]
        self.SIZE=100

    def addData(self,item):
        self.Data.append(item)

    def ProcessData(self):
        result=[]
        for i in range(len(self.Data)):
            if self.Data[i]>0:
                result.append(self.Data[i]*2)
        return result

def process_files(files):
    for f in files:
        data=open(f).read()
        print(data)

if __name__=="__main__":
    dp=dataProcessor()
    dp.addData(5)
    dp.addData(10)
    print(dp.ProcessData())
