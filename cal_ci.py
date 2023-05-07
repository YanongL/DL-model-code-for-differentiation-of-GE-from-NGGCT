'''
--置信区间计算原理:
    采用Bootstrap法的思路是：从样本数据中重复抽取1000次样本，每次抽取n例。在每个Bootstrap样本中，计算两组的中位数之差，
    最终可计算出1000个中位数之差。然后根据这1000个中位数之差，计算出它们的第2.5 百分位数和第97.5百分位数，这就是两个中
    位数之差的95%置信区间。如果该置信区间不包含0, 则可以认为两组差异有统计学意义；否则认为两组差异无统计学意义。
———主要对外函数接口:
    
#计算多个算法不同结果的指标，并绘制pr和roc曲线，把指标和曲线分别存成： save_f+'.csv',save_f+'_pr.png',save_f+'_roc.png'
def get_class_metrix(name_list,gt_list,pre_list,save_f='test'):
#计算单次预测结果的指标和置信区间,get_matrix 可以替换成任何根据gt(真实标签列表)和pre(预测概率列表)的指标函数
def get_ci_with_metric(gt,pre,matrix_fun=get_matrix):
'''
import random
import numpy as np
import warnings
import pandas as pd
from sklearn import metrics
import matplotlib
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)
##############################
#获取正负样本的数量
def get_data_num(data):
    p_num=0
    n_num=0
    all_num=len(data)
    for i in range(all_num):
        if int(data[i])==0:
            n_num+=1
        else:
            p_num+=1
    return all_num,p_num,n_num
#根据标签，预测结果，预测结果的概率值生成计算的指标：敏感性，特异性，roc_auc,准确率,ppv，npv。
def get_matrix(gt,pre_logit,name='test',flag=0):
    pre=np.array(pre_logit)>0.5
    conf_m=metrics.confusion_matrix(gt,pre)
    spe = conf_m[0][0]/(conf_m[0][0])
    if len(conf_m)==1:
        sen=0
        acc=conf_m[0][0]/len(gt)
        ppv=0
        npv=1
    else:
        sen = conf_m[1][1]/((gt==1).sum())
        spe= conf_m[0][0]/((gt==0).sum())
        acc = (conf_m[0][0]+conf_m[1][1])/len(gt)
        ppv=conf_m[1][1]/(conf_m[0][1]+conf_m[1][1])
        npv=conf_m[0][0]/(conf_m[0][0]+conf_m[1][0])
    fpr,tpr,thresholds=metrics.roc_curve(gt,pre_logit)
    precision,recall,thresholds=metrics.precision_recall_curve(gt,pre_logit)
    auc=metrics.auc(fpr,tpr)
    auc1=metrics.average_precision_score(gt,pre_logit)
    if flag==1:
        plt.figure(1)
        plt.plot(fpr,tpr,label=name+'(area = %0.3f)' % auc)
        plt.figure(2)
        #plt.plot(recall,precision,label=name+'(area = %0.3f)' % auc1,linewidth=2, linestyle='-', marker='o')
        plt.plot(recall,precision,label=name+'(area = %0.3f)' % auc1)
    return spe,sen,auc,acc,auc1,ppv,npv
#将浮点数据列表转化成保留4位有效数值的列表
def num2for(l):
    return [format(float(t),'.4') for t in l]
#从测试集当中随机选取和测试集数量相同的预测结果和标签
def gen_random_list(pre,gt):
    dataset_index=[random.randint(0,len(pre)-1) for i in range(len(pre))]
    gt_new=[gt[t] for t in dataset_index]
    pre_new=[pre[t]for t in dataset_index]
    return np.array(gt_new),np.array(pre_new)
#选取置信区间
def get_std(l):
    l=sorted(l)
    return l[24],l[974]
#计算单次预测结果的指标和置信区间
def get_ci_with_metric(gt,pre,matrix_fun=get_matrix):
    '''
    输入参数：
        gt:真实标签列表
        pre:预测的结果列表
        matrix_fun:计算指标函数,可以替换成任何想要计算的指标
    输出参数:
        up_list:计算出来的各个指标的置信区间上界
        down_list:计算出来的各个指标的置信区间下界
    '''
    result_list=[]
    up_list=[]
    down_list=[]
    for i in range(1000):
        gt_new,pre_new=gen_random_list(pre,gt)
        matrix_result=matrix_fun(gt_new,pre_new)
        result_list.append(matrix_result)
    for i in range(len(result_list[0])):
        i_matrix_list=[t[i] for t in result_list]
        t_up,t_down=get_std(i_matrix_list)
        up_list.append(t_up)
        down_list.append(t_down)

    return up_list,down_list

###########################################
#计算多个算法不同结果的指标，并绘制pr和roc曲线，把指标和曲线分别存成： save_f+'.csv',save_f+'_pr.png',save_f+'_roc.png'
def get_class_metrix(name_list,gt_list,pre_list,save_f='test'):
    '''
    输入参数:
        name_list:定义不同分类结果的标识符，各个不同的模型的都有各自的标识符
        gt_list:真实标签，输入的list的结果跟模型的名称对应上；
        pre_list:不同方法的预测结果，各个方法的结果对应放在list的一个元素
        save_f:指标的存储路径
    输出结果：
        save_f+.csv:存放计算出来的指标结果
        save_f+_roc.png:存放roc曲线
        save_f+_pr.png:存放pr_roc曲线
    '''
    ci_values=[]
    p_down_values=[]
    p_up_values=[]
    p_nums=[]
    n_nums=[]
    for i in range(len(name_list)):
        all_num,p_num,n_num=get_data_num(gt_list[i])
        p_nums.append(p_num)
        n_nums.append(n_num)
        t=get_matrix(np.array(gt_list[i]),pre_list[i],name=name_list[i],flag=1)
        #ci_t=get_result(gt_list[i],pre_list[i])
        t_down,t_up=get_ci_with_metric(gt_list[i],pre_list[i])
        ci_values.append(num2for(t))
        p_down_values.append(num2for(t_down))
        p_up_values.append(num2for(t_up))
    p_up_values=np.array(p_up_values)
    p_down_values=np.array(p_down_values)
    ci_values=np.array(ci_values)
    df=pd.DataFrame({
        'name':name_list,
        'p_num':p_nums,
        'n_num':n_nums,
        'spe':ci_values[:,0],
        'spe_l':p_down_values[:,0],
        'spe_u':p_up_values[:,0],
        'sen':ci_values[:,1],
        'sen_l':p_down_values[:,1],
        'sen_u':p_up_values[:,1],
        'ppv':ci_values[:,5],
        'ppv_l':p_down_values[:,5],
        'ppv_u':p_up_values[:,5],
        'npv':ci_values[:,6],
        'npv_l':p_down_values[:,6],
        'npv_u':p_up_values[:,6],
        'roc_auc':ci_values[:,2],
        'roc_auc_l':p_down_values[:,2],
        'roc_auc_u':p_up_values[:,2],
        'pr_auc':ci_values[:,4],
        'pr_auc_l':p_down_values[:,4],
        'pr_auc_u':p_up_values[:,4],
        'acc':ci_values[:,3],
        'acc_l':p_down_values[:,3],
        'acc_u':p_up_values[:,3],
        })
    df.to_csv(save_f+'.csv',index=False)
    plt.figure(1,figsize=(40,20),dpi=30)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic Curve')
    plt.title('ROC Curve')
    plt.legend(loc="lower right",fontsize=7.5)
    plt.savefig(save_f+'_roc.png',dpi=1000)
    plt.clf()
    plt.figure(2,figsize=(40,20),dpi=30)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(fontsize=7.5)
    plt.savefig(save_f+'_pr.png',dpi=1000)
    plt.clf()
    #plt.close()
#测试样例
if __name__=='__main__':
    name_list=['test1','test2']
    test_gt=[0,0,0,1,1,1]
    pre_test1=[0.1,0.2,0.3,0.7,0.8,0.9]
    pre_test2=[0.1,0.6,0.3,0.4,0.8,0.9]
    gt_list=[test_gt,test_gt]
    pre_list=[pre_test1,pre_test2]
    save_f='./ci_test/test'
    mkdir(os.path.split(save_f)[0])
    get_class_metrix(name_list,gt_list,pre_list,save_f)