import sys
print(sys.version)
print("hello, world!")
import random
random.randint(1, 10)
if 5 > 2:
    print("Five is greater than two!")
x = 5
y = "hello, world!"
# This is a comment
"""
This is a comment writen in multiple lines
"""
x = str(3)
y = int(3)
z = float(3)
print(type(x))

fruits = ("apple", "banana", "cherry")
x, y, z = fruits

x, y, z = "orange", "banana", "cherry"
x = y = z = "orange"
y = str(1)
x = "Python is awesome"
print(x)
print(x + y + z)
print(x, y, z)
x = "awesome"
def myfunc():
    global x
    x = "hello"
    print("Python is " + x)
myfunc()
type(x)
x = bool(5)
x = bytes(5)
x = float(1)
x + 1e-32
y = 1e-32
x += y
y = 3.6
int(y)
import random
print(random.randrange(1, 10))
for x in z:
    print(x)
a = "hello, world!"
print(a[1])
for i in range(0, 5):
    print(a[i])
print(a[2 : 5])
print(a[-5 : -1])
price = 25
txt = f"The price is {price : .2f} dollars"
print(txt)
x = 200
print(isinstance(x, int))
y = x
x is y
x
y = 3
x is y
x = [1, 3, 4]
y = x
y is x
y[2] = 1
y is x
y = x.copy()
y is x
thislist = ["apple", "banana", "cherry", "orange",
            "kiwi", "melon", "mango"]
print(thislist[-4 : -1])

a = ("a", "b", "c", "d", "e", "f", "g", "h")
x = slice(3, 5)
print(a[x])
print(a[3 : 5])
thislist = ["apple", "banana", "cherry"]
thislist[1:2] = ["blackcurrant", "watermelon"]
print(thislist)

thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["watermelon"]
print(thislist)
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
print(thislist)
tropical = ["mango", "pineapple", "papaya"]
thislist.extend(tropical)
thislist.remove("banana")
thislist.pop(2)
del thislist[0]
del thislist
thislist = ["apple", "banana", "cherry"]
thislist.clear()
thislist = ["apple", "banana", "cherry"]
newlist = [x.upper() for x in thislist if "a" in x]
newlist.sort(reverse = True)
def myFunc(n):
    return abs(n - 50)
thislist = [100, 50, 65, 82, 23]
thislist.sort(key = myFunc)
mylist = thislist.copy()
mylist = list(("hello", "world"))
newlist = mylist[:]
clist = mylist + newlist
clist.extend(clist)

thistuple = ("apple", "banana", "cherry", "apple", "cherry")
print(thistuple)
type(thistuple)
thistuple = ("apple", "banana", "cherry")
print(thistuple[1])
thistuple = ("apple", "banana", "cherry", "orange",
             "kiwi", "melon", "mango")
print(thistuple[: 4])

x = ("apple", 'banana', "cherry")
y = list(x)
y[1] = "kiwi"
x = tuple(y)
fruits = ("apple", "banana", "cherry")
(green, yellow, red) = fruits
fruits = ("apple", "banana", "cherry", "strawberry", "raspberry")
(green, yellow, *red) = fruits
print(green)
print(yellow)
print(red)
(green, *tropic, red) = fruits
print(green)
print(tropic)
print(red)

for i in range(len(thistuple)):
    print(thistuple[i])

fruits = ("apple", "banana", "cherry")
mytuple = fruits * 2
print(mytuple)
thisset = {"banana", "apple", "cherry", "apple"}
print(thisset)
for x in thisset:
    print(x)
thisset.add("orange")
thisset.remove("apple")
thisset = {"apple", "banana", "cherry"}

x = thisset.pop()
print(x)
print(thisset)
thisset.clear()
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = set1.union(set2)
thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
}
print(thisdict)
print(thisdict["brand"])
thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964,
    "year": 2020
}
len(thisdict)
thisdict["color"] = ["red", "white", "blue"]
thisdict.keys()
car = {
"brand": "Ford",
"model": "Mustang",
"year": 1964
}
x = car.keys()
print(x)
car["color"] = "white"
print(x)
id(x)
id(car)
x = car.values()
print(x)
car["year"] = 2018
print(x)
x = car.items()
if "model" in thisdict:
    print("model is one of the keys")
print(car)
car.pop("model")
car.clear()
car
for x in thisdict:
    print(x)
for x in thisdict:
    print(thisdict[x])
for x, y in thisdict.items():
    print(x, y)
mydict = thisdict.copy()
child1 = {
  "name" : "Emil",
  "year" : 2004
}
child2 = {
  "name" : "Tobias",
  "year" : 2007
}
child3 = {
  "name" : "Linus",
  "year" : 2011
}

myfamily = {
  "child1" : child1,
  "child2" : child2,
  "child3" : child3
}
myfamily["child1"]["name"]
child1["name"] = "bill"
for x in myfamily.values():
    for y in x.values():
        print(y)
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}

set3 = set1.union(set2)
set2.remove(1)
print(set3)

tuple1 = ("a", "b", "c")
tuple2 = (1, 2, 3)
tuple3 = tuple1 + tuple2
tuple2 = (3, 4, 5)
a = 330
b = 350
print("A") if a > b else print("=") if a == b else print("B")
i = 1
while i < 6:
    print(i)
    i +=1

i = 1
while i < 6:
    print(i)
    if i == 3:
        break
    i += 1
i = 0
while i < 6:
    i += 1
    if i == 3:
        continue
    print(i)
i = 1
while i < 6:
    print(i)
    i +=1
else:
    print(f"i is equal to {i : 2f}\n")
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)

for x in range(6):
    print(x)
adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]
for x in adj:
    for y in fruits:
        print(x, y)
def my_function():
    print("Hello from a function!\n")
my_function()
def my_function(**kid):
    print("His last name is " + kid["lname"])
my_function(fname = "Tobias", lname = "Refsnes")

def tri_recursion(k):
  if(k > 0):
    result = k + tri_recursion(k - 1)
    print(result)
  else:
    result = 0
  return result

print("Recursion Example Results:")
tri_recursion(6)

add10 = lambda a : a + 10
print(add10(5))
mult = lambda a, b : a * b
mult(10, 11)
def mult_n(n):
    def multn(k):
        return n * k
    func = multn
    return func

mult10 = mult_n(10)
mult10(13)

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self):
        return f"{self.name}({self.age})"
    def greeting(self):
        print("Hello, my name is " + self.name)
p1 = Person("John", 36)
print(p1)
p1.greeting()
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)
class Student(Person):
    def __init__(self, fname, lname, student_id):
        self.id = student_id
        super().__init__(fname, lname)

student1 = Student("Hongda", "Zhang", "0011")
fruits = ("apple", "banana", "cherry")
iter_fruits = iter(fruits)
next(iter_fruits)
next(iter_fruits)
next(iter_fruits)

class MyIter:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        self.current = 0
        return self
    def __next__(self):
        if self.current < self.n:
            current = self.current
            self.current +=1
            return current
        else:
            raise StopIteration
n_times = MyIter(10)
for x in n_times:
    print(x)

class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    def move(self):
        pass

class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand, model)
    def move(self):
        print("Move!")
class Boat(Vehicle):
    def move(self):
        print("Sail!")

class Plane(Vehicle):
    def move(self):
        print("Fly!")
car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Turig 20")
plane1 = Plane("Boeing", "747")
for x in (car1, boat1, plane1):
    x.move()
import platform
x = dir(platform)
print(x)
import datetime
x = datetime.datetime.now()
import json

# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])
# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

import re

txt = "The rain in Spain"
x = re.findall("[a]{1}", txt)
print(x)
del x
try:
    print(x)
except:
    print("An exception occured!")

try:
    print(x)
except NameError:
    print("Variable x is not defined!")
except:
    print("Something else went wrong!")

try:
    print("Hello")
except:
    print("Something went wrong!")
else:
    print("Nothing went wrong!")

#x = -1
#if x < 0:
#    raise Exception("Sorry, no numbers below zero")

x = "hello"

#if not type(x) is int:
#    raise TypeError("Only integers are allowed!")
username = input("Enter username:")
print("Username is: " + username)
price = 59.577574
txt = f"The price is {price : .2f} dollars!"
print(txt)

f = open("myfile.txt", "w")
f.close()
f = open("myfile.txt", "r")
print(f.readline())
f.close()

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
type(arr)
print(np.__version__)
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(f"2nd element on 1st row: {arr[0, 1]}")
arr = np.array([1, 2, 3, 4, 5, 6, 7])
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype("i")
x = arr.copy()
arr[0] = 42
print(arr)
print(x)
x = arr.view()
arr[0] = 23
print(arr)
print(x)
x = arr.copy()
y = arr.view()
x.base
y.base
arr = np.array([ range(1, 13) ])
newarr = arr.reshape(4, 3)
newarr.base

arr = np.array([[1, 2, 4], [4, 5, 6]])
for x in arr:
    print(x)
    break
for x in range(0, 2):
    for y in range(0, 3):
        print(arr[x, y])
for x in arr:
    for y in x:
        print(y)

arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x)
arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
  print(idx, x)


arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)
arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=0)

print(arr)
arr = np.concatenate((arr1, arr2), axis=1)

print(arr)

arr = np.concatenate((arr1, arr2), axis=0)

print(arr)

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)

import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)

arr = np.vstack((arr1, arr2))
y = arr[:, 0]
print(arr)

arr = np.array([1, 2, 3, 4, 5, 6, 4, 4])
newarr = np.array_split(arr, 3)
x = np.where(arr == 4)
x[0][0]

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)

print(x)

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]

newarr = arr[x]

arr = np.array([1, 2, 3, 4, 5, 6, 7])

newarr = arr[ arr % 2 == 0]

print(newarr)

from numpy import random

x = random.randint(100, size = (2, 5))

from numpy import random
x = random.choice([3, 5, 7], p = [.1, .3, .6], size = 100)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

random.shuffle(arr)

print(arr)
print(random.permutation(arr))
import seaborn as sns
import matplotlib.pyplot as plt
x = random.normal(loc = 10, scale = 2, size = (1000))
sns.kdeplot(x)
plt.show(block = True)
x = random.binomial(n = 10, p = 0.5, size = 100)
sns.displot(x)
plt.show(block = True)

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []
c = zip(x, y)
for i, j in zip(x, y):
    z.append(i + j)
print(z)

z = np.add(x, y)
print(z)

def myadd(x, y):
    return x + y
myadd = np.frompyfunc(myadd, 2, 1)

print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.add(arr1, arr2)

from math import log
nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))
newarr = np.add(arr1, arr2)
np.sum([arr1, arr2])
[arr1, arr2]
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
newarr = np.sum([arr1, arr2], axis = 0)
np.prod(arr1)
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

newarr = np.prod([arr1, arr2], axis=1)

arr = np.array([3, 6, 9])

x = np.lcm.reduce(arr)

print(x)

import pandas as pd

df = pd.read_csv("data/house_tiny.csv")

mydataset = {
    'cars' : ["BMW", "Volvo", "Ford"],
    'passings' : [3, 7, 2]
}

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

data = pd.read_json("data/data.js")

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)
df.head(10)
print(df)
df.dropna(inplace = True)
df.fillna(130, inplace = True)
df["Duration"] = df["Duration"].fillna(130)
df.index
df.plot()
plt.show(block = True)

import matplotlib.pyplot as plt
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show(block = True)

plt.plot(xpoints, ypoints, "o")
plt.show(block = True)

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints, marker = "X")
plt.show(block = True)

plt.plot(ypoints, "o:r", ms = 20)
plt.show(block = True)

plt.plot(ypoints, lw = "20", ls = "dashed", ms = 20)
plt.show(block = True)

y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])
plt.plot(y1)
plt.plot(y2)
plt.ylabel("Calorie Burnage")
plt.show(block = True)

plt.subplot(2, 1, 1)
plt.plot(y1, y2, "o:g")
plt.subplot(2, 1, 2)
plt.plot(y2, y1, "o--r")
plt.show(block = True)

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y, color = "red")
plt.show(block = True)

plt.hist(y)
plt.show(block = True)

np.mean(y)
np.std(y)
np.percentile(y, 90)
x = np.random.uniform(0, 5, 250)
plt.hist(x, 5)
plt.show(block = True)
from numpy import random
x = random.normal(5, 1, 10000)
plt.hist(x, 20)
plt.show(block = True)

import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def line(x):
    return slope * x + intercept
linear_line = list(map(line, x))
plt.scatter(x, y)
plt.plot(x, linear_line)
plt.show(block = True)
######################################################
import torch, numpy
from dask.dataframe.dispatch import tolist
from docutils.nodes import legend, inline

x = torch.arange(12)
x
x.shape
x = x.reshape(3, 4)
torch.randn(3, 4)
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 4])
x + y
x = torch.arange(12, dtype = torch.float32).reshape(3, 4)
y = torch.randn(12, dtype = torch.float32).reshape(3, 4)
z = torch.cat((x, y), dim = 0)
o = torch.cat((x, y), dim = 1)

x == y
x.sum()
x[1 : 3]
id(y)
y = x + y
id(y)
z = torch.zeros_like(y)
id(z)
z[:] = x + y
id(z)
id(x)
x += y
id(x)
x = y
id(x)
id(y)
z = x
id(z)
a = x.numpy()
type(a)
import os
os.makedirs('data', exist_ok = True)
data_file = os.path.join('data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('2,NA,106000\n')
    f.write('NA,Pave,127500\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd
data = pd.read_csv(data_file)
print(data)
inputs, mid, outputs = data.iloc[:, 0:1], data.iloc[:, 1], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
mid = pd.get_dummies(mid, dummy_na = True, dtype = 'int')
inputs = pd.DataFrame.join(inputs, mid)
x = torch.tensor(inputs.values, dtype = torch.float32)
y = torch.tensor(outputs.values, dtype = torch.float32)
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x ** y, x * y, x / y
x = torch.arange(3)
len(x)
x.shape
A = torch.arange(20).reshape(5, 4)
A[2, 3]
A.T
x = torch.arange(24).reshape(2, 3, 4)
A = torch.arange(20, dtype = torch.float32).reshape(5, 4)
B = A.clone()
id(A)
id(B)
A * B
C = A
A * C
A.sum(axis = 0)
torch.sum(A[:, 0])
A.mean()
A.sum() / A.numel()
A.mean(axis = 0), A.sum(axis = 0) / A.shape[0]
A.cumsum(axis = 0)
y = torch.ones(4, dtype = torch.float32)
x = torch.arange(4, dtype = torch.float32)
torch.dot(x, y)
A.shape
torch.mv(A, y)
B = torch.randn(4, 3)
torch.mm(A, B)
torch.norm(x)
torch.abs(x).sum()
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib_inline import backend_inline
def f(x):
    return 3 * x ** 2 - 4 * x
f(11)

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h = {h : .5f}, numerical limit = {numerical_lim(f, 1, h) : .5f}')
    h *= .1

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def has_one_axis(X):
    return (hasattr(X, 'ndim') and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], '__len__'))

def plot(X, Y = None, xlabel = None, ylabel = None, legend = None,
         xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
         fmts = ('-', 'm--', 'g-', 'r:'), figsize = (35, 25), axes = None):
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        plt.show(block = True)
x = np.arange(0, 3, 0.1)
plt.rcParams.update({'font.size': 22}) # change font size
plot(x, [f(x), 2 * x -3], 'x', 'f(x)', legend = ['f(x)', 'Tangent line(x = 1)'])

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)
plt.show(block = True)
x = torch.arange(4.0, dtype = torch.float32)
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x, x)
y.backward()
x.grad
x.grad == 4 * x
y = x.sum()
x.grad.zero_()
y.backward()
x.grad
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u
x.grad.zero_()
v = y.clone()
z = v * x
z.sum().backward()
x.grad == v

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size = (), requires_grad = True)
d = f(a)
d.backward()
a.grad == d / a
a.grad.zero_()

torch.ones(4, dtype = torch.float32)


import math
import time
import numpy as np
import torch
n = 10000
a = torch.ones(n)
b = torch.ones(n)
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum(),tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'

timer.start()
d = a + b
f'{timer.stop():.5f} sec'

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
y = [normal(x, mu, sigma) for mu, sigma in params]
plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel = "x",
     ylabel = 'PDF', figsize = ( 45, 25), legend = [f'mean {mu}, std {sigma}'
                                                     for mu , sigma in params])
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y +=torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show(block = True)
import random
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
w = torch.normal(0, 0.01, size = (2, 1), requires_grad= True)
b = torch.zeros(1, requires_grad = True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
w = torch.normal(0, 0.01, size = (2, 1), requires_grad= True)
b = torch.zeros(1, requires_grad = True)
lr = .01
num_epochs = 10
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()): f}")
true_w - w.detach().reshape(true_w.shape)
true_b - b.detach()
import numpy as np
import torch
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, b, 1000)
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

for X, y in data_iter:
    print(X)
    print(y)
    break

next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l : f}")

   w = net[0].weight.data
   true_w - w.reshape(true_w.shape)
   b = net[0].bias.data
   true_b - b

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
use_svg_display()
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root = ".data", train = True, transform = trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(
    root = "data", train = False, transform = trans, download = True
)

len(mnist_train), len(mnist_test)
mnist_train[0][0].shape

def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal",
                   "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]
labels = range(1, 10)
get_fashion_mnist_labels(labels)

def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if(titles):
            ax.set_title(titles[i])
    plt.show(block = True)
    return(axes)
X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18, 28, 28), 2, 9,
            titles = get_fashion_mnist_labels(y))
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")

x = zip(a, b)
class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration
myclass = MyNumbers()
myiter = iter(myclass)
for x in myiter:
    print(x)