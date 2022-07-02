# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:11:17 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D  #用于支持3D图
from scipy.interpolate import lagrange


#导入数据
filepath = r"C:/Users/lenovo/Desktop/Typhoon/SURF_CLI_CHN_MUL_DAY_RHU_Wuhan_5_stations.xlsx"

data = pd.read_excel(io = filepath, sheet_index=0, header=0)

#拉格朗日插值
def ployinterp_column(s, n, k = 2):   #插值函数
    y = s[list(range(n-k-1,n-1))+list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index,list(y))(n)

for j in data.index:
    if(data['RHUavg'].isnull())[j]:
        data['RHUavg'][j] = ployinterp_column(data['RHUavg'], j) #拉格朗日插值
print(sum(data['RHUavg'].isnull()))  #确认插值完成

#
#分割数据集
def year_select(data, n):
    data1 = list()
    for j in range(0, data.shape[0]):
        if(data['year'][j] == n):
            if(data['month'][j]==2 and data['day'][j]==29):
                pass #去掉2月29日
            else:
                data1 = np.append(data1, data['RHUavg'][j])#找到对应年份
    return data1


data2 = year_select(data, 1960)
for i in range(1961,1967):  #1968年出现四个月的缺失，暂时舍去
    data2 = np.column_stack((data2, year_select(data, i)))
for u in range(1969,2017):
    data2 = np.column_stack((data2, year_select(data, u)))

#求出单日均值的向量
mean_data2 = np.mean(data2, axis = 1) #日平均，需要对行求和

#使用傅里叶基对函数轨道进行恢复
def fourier_fit(data, para = 250, graph = False):
    n = len(data)
    def fourier_curve(x, *a):
        w = 2 * np.pi / n
        ret = 0
        for deg in range(0, int(len(a) / 2) + 1):
            ret += a[deg] * np.cos(deg * w * x) + a[len(a) - deg - 1] * np.sin(deg * w * x)
        return ret
    x = np.arange(1, n+1, 1)
    y = np.array(data)
    popt, pcov = curve_fit(fourier_curve, x, y, [1.0]*para)
    if (graph == True):
        plt.plot(x, y, 'o',color = 'r', markersize = 2.0, label = "Initial Data")
        plt.plot(x, fourier_curve(x, *popt), color = 'g', label = "Fit Function")
        plt.legend()
        plt.show()
    return popt

mean_para = fourier_fit(mean_data2, graph = True)
fourier_fit(data2[:,0],graph = True)
#data_f作为函数基的参数，其相当于还原了函数轨道，函数型数据分析可以基于此展开。
import warnings
data_f = fourier_fit(data2[:,0]-mean_data2)
for p in range(1,55):
    warnings.filterwarnings('ignore')
    data_f =  np.column_stack((data_f, fourier_fit(data2[:,p]-mean_data2)))

#对矩阵进行三维可视化
def draw_matrix(matrix):
    from mpl_toolkits.mplot3d import Axes3D  #用于支持3D图
    size = matrix.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X, Y, matrix)
    plt.show()

draw_matrix(data2)

#提取函数在1到365的取值，用于计算协方差并进行主成分基的求解
def fourier_curve1(x, a):
        w = 2 * np.pi / 365
        ret = 0
        for deg in range(0, int(len(a) / 2) + 1):
            ret += a[deg] * np.cos(deg * w * x) + a[len(a) - deg - 1] * np.sin(deg * w * x)
        return ret #单次只能返回一个值


##重新从函数中提取对应的观测点
X_pick = list(map(lambda x: fourier_curve1(x, data_f[:,0]), np.arange(0,365,1)))
for t in range(1, 55):
    X_pick = np.column_stack((X_pick, list(map(lambda x: fourier_curve1(x, data_f[:,t]), np.arange(0,365,1)))))
draw_matrix(X_pick)
draw_matrix(data2)



#主成分分析
COV_year = np.cov(np.transpose(X_pick)) #不同年份为不同的函数轨道
COV_day = np.cov(X_pick)
eig_values, eig_function = np.linalg.eig(COV_day) #未对特征值进行大小排序
idx = eig_values.argsort()[::-1] #重新按照降序排序
eig_values = eig_values[idx]
eig_function = eig_function[:,idx]

plt.plot(eig_values/sum(eig_values).real, color = 'r', label = 'CR')
plt.plot(np.cumsum(eig_values/sum(eig_values).real), label = 'ACR')  #可以考虑取53个主成分基函数
plt.legend()
plt.show()

np.cumsum(eig_values/sum(eig_values)).real


#将特征向量恢复为特征基函数，使用傅里叶基进行恢复
fourier_fit(np.array(eig_function[:,0].real), graph = True)

eigen_function = fourier_fit(np.array(eig_function[:,0].real))
for u in range(1,70):#保留53个主成分基函数，累计贡献率99%
    eigen_function = np.column_stack((eigen_function, fourier_fit(np.array(eig_function[:,u].real))))

#提取主成分基函数的对应观测值，用于生成随机数
eigen_pick = list(map(lambda x: fourier_curve1(x, eigen_function[:,0].real), np.arange(0,365,1)))
for t in range(1, 70):
    eigen_pick = np.column_stack((eigen_pick, list(map(lambda x: fourier_curve1(x, eigen_function[:,t].real), np.arange(0,365,1)))))
score02 = np.dot(np.transpose(X_pick), eigen_pick)

#简化版：使用特定分布拟合
import seaborn as sns
sns.kdeplot(score02[:,16])   #核密度估计
sns.kdeplot(score02[:,16], cumulative = True)   #累计核密度

from scipy import stats
test = score02[:,20]
stats.kstest(test, 'norm', args=(test.mean(),test.std())) #检验是否为正态分布

#生成随机数代替主成分得分
para_nu = np.random.normal(score02[:,0].mean(),score02[:,0].std()+0.105,1)
for t in range(1,70):
    para_nu = np.row_stack((para_nu, np.random.normal(score02[:, t].mean(),score02[:, t].std()+0.105,1)))

mean_pick = fourier_curve1(np.arange(0, 365, 1), mean_para)
eigen_recover = fourier_curve1(np.arange(0, 365, 1), eigen_function[:,0].real)
for q in range(1, 70):
    eigen_recover = np.column_stack((eigen_recover, fourier_curve1(np.arange(0, 365, 1), eigen_function[:,q].real)))

v4 = np.dot(eigen_recover, para_nu)
result = np.transpose(np.array(mean_pick + np.transpose(v4)))
result1 = mean_pick + np.dot(eigen_recover, para_nu)
for d in range(0,len(result)):
    if (result[d]>=1):
        result[d]=1
    if (result[d]<0):
        result[d]=0
plt.plot(mean_pick + np.dot(eigen_recover, para_nu))



sns.kdeplot(data2[:,40])   #核密度估计
sns.kdeplot(result[:,0], cumulative = True)   #累计核密度
test = data2[:,40]
result01 = result[:,0]

#将数据与2000年的相关数据进行对比，分布特征如下：
#生成散点
plt.plot(result,'o',markersize = 2.0,label = 'RHUavg_simulate')
plt.plot(data2[:,40],'o',markersize = 2.0, color = 'r', label = 'RHUavg_2000')
plt.legend()
plt.show()

#生成时序
plt.plot(result,label = 'RHUavg_simulate')
plt.plot(data2[:,40], color = 'r', label = 'RHUavg_2000')
plt.legend()
plt.show()

#核密度对比
sns.kdeplot(np.transpose(result01),label = 'RHUavg_simulate')
sns.kdeplot(test, label = 'RHUavg_2000')
plt.legend()
plt.show()

sns.kdeplot(np.transpose(result01), cumulative = True,label = 'RHUavg_simulate')
sns.kdeplot(test, cumulative = True, label = 'RHUavg_2000')
plt.legend()
plt.show()

np.mean(result01)
np.mean(data2[:,40])


np.std(result01)
np.std(data2[:,40])



#生成散点
plt.plot(result,'o',markersize = 2.0,label = 'RHUavg_simulate')
plt.plot(data2[:,20],'o',markersize = 2.0, color = 'r', label = 'RHUavg_1980')
plt.legend()
plt.show()

#生成时序
plt.plot(result,label = 'RHUavg_simulate')
plt.plot(data2[:,20], color = 'r', label = 'RHUavg_1980')
plt.legend()
plt.show()

#核密度对比
sns.kdeplot(result[:,0],label = 'RHUavg_simulate')
sns.kdeplot(data2[:,20], label = 'RHUavg_1980')
plt.legend()
plt.show()

sns.kdeplot(np.transpose(result[:,0]), cumulative = True,label = 'RHUavg_simulate')
sns.kdeplot(data2[:,20], cumulative = True, label = 'RHUavg_1980')
plt.legend()
plt.show()


#改进：随机数生成
plt.plot(score02[:,6])











