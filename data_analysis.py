import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from scipy.optimize import curve_fit
from matplotlib.cm import get_cmap

n_participants=5
means=[36,37,40,42]
means_1=[36,40,50,54]
__dir__ = os.path.abspath(os.path.dirname(__file__))
original_palette = sns.color_palette("muted", n_colors=3)  # 原始 Pastel1 调色板
# dark_palette = sns.dark_palette(original_palette[0], n_colors=3)  # 生成深色调

def curve(start_index,end_index,color='red',legend='stage1',outlier=False):
    plt.figure(figsize=(8, 6))
    d12=pd.DataFrame()
    d13=pd.DataFrame()
    d14=pd.DataFrame()
    d23=pd.DataFrame()
    d24=pd.DataFrame()
    d34=pd.DataFrame()
    for i in range(n_participants):
        data_path=os.path.join(__dir__,'data',str(i+1)+'.csv')
        data=pd.read_csv(data_path,index_col=None)
        data=data.iloc[start_index:end_index]
        d12=pd.concat([d12,data[(data['cor_option']==2) & (data['inc_option']==1)][['rt','accuracy']]],ignore_index=True)
        d13=pd.concat([d13,data[(data['cor_option']==3) & (data['inc_option']==1)][['rt','accuracy']]],ignore_index=True)
        d14=pd.concat([d14,data[(data['cor_option']==4) & (data['inc_option']==1)][['rt','accuracy']]],ignore_index=True)
        d23=pd.concat([d23,data[(data['cor_option']==3) & (data['inc_option']==2)][['rt','accuracy']]],ignore_index=True)
        d24=pd.concat([d24,data[(data['cor_option']==4) & (data['inc_option']==2)][['rt','accuracy']]],ignore_index=True)
        d34=pd.concat([d34,data[(data['cor_option']==4) & (data['inc_option']==3)][['rt','accuracy']]],ignore_index=True)
    
    x=np.array([math.log(36/54),math.log(36/50),math.log(40/54),math.log(40/50),math.log(36/40),math.log(50/54),math.log(54/50),math.log(40/36),math.log(50/40),math.log(54/40),math.log(50/36),math.log(54/36)])
    # 顺序：14,13,24,23,12,34,43,21,32,42,31,41
    p12=(d12['accuracy']==1.0).sum()/d12.count()
    p13=(d13['accuracy']==1.0).sum()/d13.count()
    p14=(d14['accuracy']==1.0).sum()/d14.count()
    p23=(d23['accuracy']==1.0).sum()/d23.count()
    p24=(d24['accuracy']==1.0).sum()/d24.count()
    p34=(d34['accuracy']==1.0).sum()/d34.count()
    rt12=d12['rt'].mean()
    rt13=d13['rt'].mean()
    rt14=d14['rt'].mean()
    rt23=d23['rt'].mean()
    rt24=d24['rt'].mean()
    rt34=d34['rt'].mean()
    std12=d12['rt'].std()
    std13=d13['rt'].std()
    std14=d14['rt'].std()
    std23=d23['rt'].std()
    std24=d24['rt'].std()
    std34=d34['rt'].std()
    f=pd.DataFrame()
    y=np.array([1-float(p14.iloc[0]),1-float(p13.iloc[0]),1-float(p24.iloc[0]),1-float(p23.iloc[0]),1-float(p12.iloc[0]),1-float(p34.iloc[0]),float(p34.iloc[0]),float(p12.iloc[0]),float(p23.iloc[0]),float(p24.iloc[0]),float(p13.iloc[0]),float(p14.iloc[0])])
    y1=np.array([rt14,rt13,rt24,rt23,rt12,rt12,rt23,rt24,rt13,rt14])
    yerr=np.array([std14/4,std13/4,std24/4,std23/4,std12/4,std12/4,std23/4,std24/4,std13/4,std14/4])
    if outlier==True:
        x_o=np.array([math.log(36/54),math.log(36/50),math.log(40/54),math.log(40/50),math.log(36/40),math.log(50/54),math.log(54/50),math.log(40/36),math.log(50/40),math.log(54/40),math.log(50/36),math.log(54/36)])
        y_o=np.array([rt14,rt13,rt24,rt23,rt12,rt34,rt34,rt12,rt23,rt24,rt13,rt14])
        yerr_o=np.array([std14/4,std13/4,std24/4,std23/4,std12/4,std34/4,std34/4,std12/4,std23/4,std24/4,std13/4,std14/4])
    # print(std12,std13,std14,std23,std24,std34)
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.scatter(x,y1,color='red',marker='o')
    # 1. 定义高斯函数
    def gaussian(x, a, b, c):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    def lorentzian(x,a,b,c):
        return a / (1 + ((x - b) / c) ** 2)
    # x_data = np.linspace(-0.6, 0.6, 25)
    # a, b, c = 2.0, 0.0, 0.2  # 实际参数
    # y_data = gaussian(x_data, a, b, c) + 0.1 * np.random.normal(size=x_data.size)
    # initial_guess = [2, 0, 0.1]  # 初始猜测值
    # popt, pcov = curve_fit(gaussian, x, y1, p0=initial_guess)
    
    # if outlier==False:
    #     plt.scatter(x,y1,color=color,marker='o')
    #     plt.errorbar(x,y1,yerr=yerr,fmt='o')
    # else:
    #     plt.scatter(x_o,y_o,color=color,marker='o')
    #     plt.errorbar(x_o,y_o,yerr=yerr_o,fmt='o')
    # sns.lineplot(x=x_data, y=gaussian(x_data, *popt), label=legend, color=color,linestyle='-')
    plt.scatter(x,y,color='red',marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Values  A:B')
    plt.savefig(legend+'.png')
    # plt.legend(title='experiment stage',loc='best')

curve(0,30,original_palette[0],'0-30',outlier=False)
curve(30,60,original_palette[1],'30-60',outlier=False)
curve(60,90,original_palette[2],'60-90',outlier=True)
# plt.savefig('5.png')
