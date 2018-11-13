
import torch
import torch.nn.functional as F

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

loss3 = (x*y - (x*x*y).detach())**2
loss3.backward()
print ("Loss3: ", x.grad) #-48
x.grad = None

loss4 = (x*x*y - (x*y).detach())**2
loss4.backward()
print ("Loss4: ", x.grad) #288
x.grad = None

loss5 = (x*x*y - x*y)**2
loss5.backward()
print ("Loss5: ", x.grad) #240
x.grad = None

loss5 = (x*y - x*x*y)**2
loss5.backward()
print ("Loss6: ", x.grad) #240
x.grad = None