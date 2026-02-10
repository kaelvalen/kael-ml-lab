import os
import pandas as pd
import torch

os.makedirs(os.path.join("mydata"), exist_ok=True)
data_file = os.path.join("mydata", "mydata.csv")
with open(data_file, "w") as f:
    f.write('''Cpu,Gpu,Ram,Price
i5-9400f,gtx1660,16gb,1500
i7-9700f,rtx2060,16gb,2000
i9-9900k,rtx2080,32gb,3000''')
    
data = pd.read_csv(data_file)
print('my data:')
print(data)

inputs, targets = data.iloc[:, :-1], data.iloc[:, -1]
inputs = pd.get_dummies(inputs, dummy_na=True)
print('inputs:')
print(inputs)
print('targets:')
print(targets) 

inputs = inputs.fillna(inputs.mean())
print('inputs after filling missing values:')
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print('X:')
print(X)
print('y:')
print(y)