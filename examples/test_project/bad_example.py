# Bad example with violations
import os,sys
from pathlib import *

def calcAvg(nums):
    return sum(nums)/len(nums)

class dataProcessor:
    def __init__(self):
        self.Data=[]
    
    def addItem(self,item):
        self.Data.append(item)

if __name__=="__main__":
    dp=dataProcessor()
    dp.addItem(5)
    print(dp.Data)
