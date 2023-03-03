#!/usr/bin/env python
# coding: utf-8

# ## Support Vector Machine

# Support Vector Machine tries to find a line/hyperplane (in multidimensional space) that separates two classes.
# - Usually for classification,can be used for regression too
# `
# SVM algorithms use a set of mathematical functions, defined as the kernel.<br>
# __Kernel:__  
# - A kernel helps us find a hyperplane in the higher dimensional space without increasing the computational cost. 
# - There are li,Gaussian and sigmoid.<br>
# - “rbf”(radial basis function) = Gaussian and “poly”(polynomial kernel) are useful for non-linear hyper-plane. It’s called nonlinear svm.
# - __gamma:__ Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid.’ The higher value of gamma will try to fit them exactly as per the training data set, i.e., generalization error and cause over-fitting problem.
# 
# - __C:__ Penalty parameter C of the error term. It also controls the trade-off between smooth decision boundaries and classifying the training points correctly,causing overfitting by shrinking the decision boundary
# 
# 
# The most used type of kernel function is RBF. It is also the default kernel.<br>
# -__Classifier:__ `sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)`

# __Pros of SVM__
# 
# - It works really well with a clear margin of separation.
# - It is effective in high-dimensional spaces.
# - It is effective in cases where the number of dimensions is greater than the number of samples.
# - It uses a subset of the training set in the decision function (called support vectors), so it is also memory efficient.

# __Cons:of SVM__
# 
# 
# - It doesn’t perform well when we have a large data set because the required training time is higher.
# - It also doesn’t perform very well when the data set has more noise, i.e., target classes are overlapping.
# - SVM doesn’t directly provide probability estimates; these are calculated using an expensive five-fold cross-validation. It is included in the related SVC method of the Python scikit-learn library.<BR>
# The algorithm works best when the number of dimensions is greater than the number of samples and is not recommended to be used for noisy, large, or complex data sets.
# 

# In[213]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR ,SVC
from sklearn import datasets


# ### Support Vector Classifier

# In[214]:


#Using the iris dataset in sklearn library

iris = datasets.load_iris()
#We only take the first two features(sepal_length ,sepal_width)
X = iris.data[:, :2]

y = iris.target


# In[215]:


#create a mesh grid to show the classificatio regions

#defining limit for x axis
x_min,x_max = X[:, 0].min() -1,X[: ,0].max() + 1
#defining limit for y axis
y_min,y_max = X[: ,0].min()-1, X[: ,1].max() + 1

h = (x_max / x_min)/100


xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

#plt.subplot(xx,yy)


# In[216]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X, y)


# In[217]:


plt.figure(figsize = (8,5),dpi = 80)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

plt.scatter(X[: , 0], X[:,1], c = y, cmap =plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(),xx.max())
plt.title("Default SVM")
plt.show()

fig.savefig("default svm")


# In[218]:


#same plot as above
plt.figure(figsize = (8,5),dpi = 80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5);

plt.scatter(X[: , 0], X[:,1], c = y, cmap =plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(),xx.max())
plt.title("Default SVM")
plt.show()

k = svc.score(X,y)
print('Training Score', k )

fig.savefig("default svm4")


# ### Support Vector Regressor

# In[197]:


data = pd.read_csv("posi")
data


# In[198]:


x = data.iloc[:,1:2].values
x


# In[199]:


y = data.iloc[:,2:].values
y


# In[200]:


#feature scaling is making all the numeric value equal.

st_x = StandardScaler()
st_y = StandardScaler()

st_x


# In[201]:


X = st_x.fit_transform(x)
y = st_y.fit_transform(y)


# In[202]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(X,y,color = "r")


# __observations__
# - Data follows a polynomial distribution

# In[203]:


from sklearn.svm import SVR


# In[204]:


regressor = SVR(kernel = "rbf")


# In[205]:


regressor.fit(X,y)


# In[211]:


plt.scatter(X,y)
plt.plot(X, regressor.predict(X),color = "purple");
k = regressor.score(X,y)
print('Trainin Score', k )


# In[207]:


regressor2 = SVR(kernel = "poly")


# In[208]:


regressor2.fit(X,y)


# In[210]:


plt.scatter(X,y)
plt.plot(X, regressor2.predict(X),color = "purple");

k = regressor2.score(X,y)
print('Training Score', k )


# In[ ]:




