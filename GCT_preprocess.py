import skimage.measure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pandas as pd
import numpy as np
import os,shutil
import warnings
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn import metrics

warnings.filterwarnings('ignore')
##############################
import SimpleITK as sitk
from cal_ci import get_class_metrix,get_matrix
##################################### T-Test########################################
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
def ttest_arr(train_x,train_y):
    train_0=[]
    train_1=[]
    for i in range(len(train_x)):
        if train_x[i]=='':
            continue
        if train_y[i]<1:
            train_0.append(train_x[i])
        else:
            train_1.append(train_x[i])
    return train_0,train_1
def get_all_result(train_f,test_f,key_list,save_f):
    train_df=pd.read_csv(train_f)
    test_df=pd.read_csv(test_f)
    names=[]
    pvalue=[]
    statistic=[]
    for k in key_list:
        print(k)
        names.append(k)
        train_0,train_1=ttest_arr(train_df[k],train_df['label'])
        test_0,test_1=ttest_arr(test_df[k],test_df['label'])
        t=ttest_ind(train_0+test_0,train_1+test_1)
        print(t[1])
        pvalue.append(t[1])
        statistic.append(t[0])
    df=pd.DataFrame({'name':names,'statistic':statistic,'pvalue':pvalue})
    df.to_csv(save_f,index=False)
def kafa_arr(train_x,train_y):
    train_set={}
    ind=0
    for t in train_x:
        if t in train_set.keys():
            continue
        train_set[t]=ind
        ind+=1
    kafa_arr=np.zeros((len(train_set),2))
    for i in range(len(train_x)):
        kafa_arr[train_set[train_x[i]],train_y[i]]+=1
    return kafa_arr,train_set
def print_kafa(test_arr,ts_dict):
    for k,v in sorted(ts_dict.items(),key=lambda d:d[0]):
        print('\t',k,int(test_arr[v].sum()),int(test_arr[v,0]),int(test_arr[v,1]))
        print('\t',' ',format(test_arr[v].sum()/test_arr.sum(),'.1%'),format(test_arr[v,0]/test_arr[:,0].sum(),'.1%'),format(test_arr[v,1]/test_arr[:,1].sum(),'.1%'))
    kf_test=chi2_contingency(test_arr)
    print(format(kf_test[1],'.3f'))
def get_kafa(train_f,test_f,k_list):
    #train_x,train_y,test_x,test_y=split_train_test(k)
    train_df=pd.read_csv(train_f)
    test_df=pd.read_csv(test_f)
    train_df=pd.concat([train_df,test_df],ignore_index=True)
    #print(train_df)
    for k in k_list:
        print(k)
        train_arr,tr_dict=kafa_arr(train_df[k],train_df['label'])
        #test_arr,ts_dict=kafa_arr(test_df[k],test_df['label'])
        #print('\ttrain')
        print_kafa(train_arr,tr_dict)
        #print('\ttest')
        #print_kafa(test_arr,ts_dict)

def read_feature(train_f,key_list):
    df=pd.read_csv(train_f)
    train_x=[]
    for i in range(len(df)):
        t_x=[]
        for t in key_list:
            t_x.append(df[t][i])
        train_x.append(np.array(t_x))
    return np.array(train_x),df['label']


##################################### dl predict ########################################
def read_dataset(path_csv):
    df=pd.read_csv(path_csv)
    return [[str(df.name[i]),df.label[i]] for i in range(len(df))]
def test_remove(data,img_path='',flag=False):
    drop_list=[]
    data_new=[t for t in data]
    for i,t in enumerate(data) :
        if not os.path.exists(os.path.join(img_path,t[0]+'.nii.gz')):
            data_new.remove(t)
            #drop_list.append(i)
            #print(t[0])
            continue
        img=sitk.ReadImage(os.path.join(img_path,t[0]))
        print(t[0])
        img_data=sitk.GetArrayFromImage(img)
        if img_data.sum()==0:
            #print(t[0])
            #drop_list.append(i)
            data_new.remove(t)
    return data_new
def test_img(data,img_path='',flag=False):
    img_list=[]
    for t in data:
        img=sitk.ReadImage(os.path.join(img_path,t[0]))
        img_data=sitk.GetArrayFromImage(img)
        img_list.append(img_data)
    return img_list
def get_max_num(data):
    t_0=[]
    t_1=[]
    t_2=[]
    for j in range(len(data[0])):
        t_0.append(np.zeros(data[0][j].shape))
        t_1.append(np.zeros(data[0][j].shape))
        t_2.append(np.zeros(data[0][j].shape))
    for i in range(len(data)):
        for j in range(len(data[0])):
            t_1[j]+=data[i][j]==1
            t_2[j]+=data[i][j]==2
    for j in range(len(data[0])):
        t_0[j][t_1[j]>t_2[j]]=1
        t_0[j][t_1[j]<t_2[j]]=2
    return t_0
def test_model(data,img_list,name='name',save_f='test_case.csv',flag=False):
    pre_masks=[]
    val_masks=[]
    pre_logit=[]
    names=[]
    pfa_nums=[]
    pfb_nums=[]
    err_l=[]
    for i in range(len(data)):
        t=data[i]
        img_data=img_list[i]
        #if img_data.sum()==0:
        #    continue
        names.append(t[0])
        pfa_num=(img_data==1).sum()
        pfb_num=(img_data==2).sum()
        pf_all=(img_data>0).sum()
        pfa_nums.append(pfa_num)
        pfb_nums.append(pfb_num)
        #pre_logit.append((pfb_num+1e-7)/(1e-7+pf_all))
        pre_logit.append((pfb_num)/(1e-7+pf_all))
        val_masks.append(t[1])
        pre_masks.append(pfb_num>pfa_num)
        if (pfb_num>pfa_num)==(t[1]>0):
            err_l.append(1)
        else:
            err_l.append(0)
    pre_list=np.array(pre_masks)
    if flag:
        df=pd.DataFrame({'name':names,'pfb_pro':pre_logit})
        df.to_csv(save_f,index=False)
    val_list=np.array(val_masks)
    logit_list=np.array(pre_logit)
    conf_m = confusion_matrix(val_list,pre_list)
    print(conf_m)
    fpr, tpr, thresholds = metrics.roc_curve(val_list, logit_list, pos_label=1)
    print(metrics.auc(fpr, tpr))
    return val_list,pre_list,logit_list
def for_mat(l):
    return [format(t,'.3g') for t in l]
def output_case_result(Tests,p_logits,p_names,save_f='test_cases.csv'):
    p_logits=np.array(p_logits)
    #all_logits=p_logits.mean(0)
    all_logits=p_logits[-1]
    Tests=np.array(Tests)
    re_dict={'name':Tests[:,0]}
    for i in range(len(p_logits)):
        re_dict[p_names[i]]=for_mat(p_logits[i])
    #re_dict['patient']=for_mat(all_logits)
    re_dict['label']=Tests[:,1]
    err_list=[(all_logits[t]>0.5)==(int(Tests[:,1][t])>0) for t in range(len(all_logits))]
    re_dict['err']=for_mat(err_list)
    df=pd.DataFrame(re_dict)
    df.to_csv(save_f,index=False)
###########################################
def cal_test_result(img_path,test_csv,save_p):
    #img_path='/hd/wjy/nnunet/nnUNet_raw/nnUNet_raw_data/Task170_GCT/preTs'
    #Tests=read_dataset('/hd/wjy/data_process/liyanong/data/Gct_test_9_8_new.csv')
    #Tests=read_dataset('../../../data/0725_test.csv')
    Tests=read_dataset(test_csv)
    mkdir(save_p)
    result_f=os.path.join(save_p,'test')
    p_img=[]
    p_logits=[]
    p_name=[]
    p_ys=[]
    for i in range(5):
        img_path_t=img_path+'_'+str(i)
        Tests=test_remove(Tests,img_path_t)
    for i in range(5):
        img_path_t=img_path+'_'+str(i)
        print(len(Tests))
        img_list=test_img(Tests,img_path_t)
        print(np.array(img_list).shape)
        y,p,logit=test_model(Tests,img_list,save_f=result_f+'fold'+str(i)+'.csv')
        p_logits.append(logit)
        p_img.append(img_list)
        p_ys.append(y)
        p_name.append('fold-'+str(i))
    p_logit_new=np.array(p_logits).max(0)
    p_img=get_max_num(p_img)
    y,p,logit=test_model(Tests,p_img,save_f=result_f+'all-voxel.csv')
    p_name.append('all-voxel')
    p_ys.append(y)
    p_logits.append(logit)
    output_case_result(Tests,p_logits,p_name,save_f=result_f+'_case.csv')
    p_logits.append(p_logit_new)
    p_ys.append(y)
    p_name.append('all-patient')
    get_class_metrix(p_name,p_ys,p_logits,result_f)
#############################################################################################
#提取的dicom文件的關鍵字
key_word=['ImageType','StudyDate','Modality','Manufacturer','StudyDescription',
          'SeriesDescription','ManufacturerModelName','PatientName','PatientID',
          'PatientSex','PatientAge','PatientWeight','MRAcquisitionType','SliceThickness',
          'MagneticFieldStrength','ProtocolName','Rows','Columns','FlipAngle',
          'RepetitionTime','EchoTime','PixelSpacing','ImagesInAcquisition']
#key_word=['PatientID','StudyDate','PatientAge','PatientSex','PatientName']
#獲取文件夾下的第一個有效dicom數據
def get_first_dcm(p):
    print(p)
    if os.path.isdir(p):
        for t in os.listdir(p):
            if 'none' in p.lower():
                continue
            try:
                data=get_first_dcm(os.path.join(p,t))
                return data
            except:
                continue
    else:
        data=pydicom.read_file(p)
        return data
    a=1/0
#新建文件夾
def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)
#把新文件路徑p下的dicom文件的信息更新到s_dict關鍵字字典裏。
def update_study_info(p,s_dict={}):
    Data=get_first_dcm(p)
    #s_dict['slice_num'].append(len(os.listdir(p)))
    s_dict['path'].append(p)
    for k in key_word:
        if hasattr(Data, k) == True:
            s_dict[k].append(Data[k].value)
        else:
            s_dict[k].append('')
    return s_dict
#將收集起來的dicom信息以csv的格式輸出到save_f文件
def write_sdict(s_dict,save_f):
    df=pd.DataFrame(s_dict)
    df.to_csv(save_f,index=False)
#初始化
def init_s_dict():
    s_dict={'path':[]}
    for k in key_word:
        s_dict[k]=[]
    #s_dict['slice_num']=[]
    return s_dict
def add_label(new_dict,label_csv):
    df=pd.read_csv(label_csv)
    for k in df.keys():
        if not k in new_dict.keys():
            new_dict[k]=['']*len(new_dict['nii_name'])
    label_dict={}
    for i in range(len(df)):
        t_dict={k:df[k][i] for k in df.keys()}
        label_dict[df['name'][i]]=t_dict
    for i,n in enumerate(new_dict['nii_name']):
        if n in label_dict.keys():
            for k in df.keys():
                new_dict[k][i]=label_dict[n][k]
            new_dict['label_path'][i]+=label_csv.split('/')[-1]
def add_client_info(new_dict,df):
    for k in df.keys():
        if not k in new_dict.keys():
            new_dict[k]=['']*len(new_dict['nii_name'])
    label_dict={}
    for i in range(len(df)):
        t_dict={k:df[k][i] for k in df.keys()}
        label_dict[df['names'][i].replace('_','').replace('1','')]=t_dict
    for i,n in enumerate(new_dict['PatientName']):
        n=n.replace(' ','').replace('^','')
        if n in label_dict.keys():
            for k in df.keys():
                new_dict[k][i]=label_dict[n][k]
            if len(new_dict['年龄'][i])>0:
                if abs(int(new_dict['年龄'][i][:-1])-int(new_dict['PatientAge'][i][:-1]))>1:
                    new_dict['备注'][i]+='age mismatch'
def add_bingli_info(new_dict,df):
    for k in df.keys():
        if not k in new_dict.keys():
            new_dict[k]=['']*len(new_dict['姓名'])
    label_dict={}
    for i in range(len(df)):
        t_dict={k:str(df[k][i]) for k in df.keys()}
        label_dict[df['姓名'][i]]=t_dict
    for i,n in enumerate(new_dict['姓名']):
        if n in label_dict.keys():
            for k in df.keys():
                if len(str(new_dict[k][i]))==0:
                    new_dict[k][i]=str(new_dict[k][i])+label_dict[n][k]
                elif not label_dict[n][k] in str(new_dict[k][i]):
                    #new_dict['备注'][i]+=(k+'mismatch')
                    new_dict[k][i]=str(new_dict[k][i])+':'+label_dict[n][k]
def split_train_test(df_normal,key_list):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    test_x_2021=[]
    test_y_2021=[]
    dataset=[]
    for i in range(len(df_normal)):
        if 'train.csv' in df_normal['label_path'][i] :
            x_t=[]
            for t in key_list:
                if ''== df_normal[t][i] or 'NA' in str(df_normal[t][i]) or '无' in str(df_normal[t][i]):
                    x_t.append(-1) 

                elif 'neg' in str(df_normal[t][i]).lower() or '＜' in str(df_normal[t][i]).replace('<','＜'):
                    x_t.append(0)
                else:
                    try:
                        x_t.append(float(df_normal[t][i]))
                    except:
                        x_t.append(-1)
            if -1 in x_t or df_normal['label(0=GE,1=NGGCT)'][i]=='NA':
                dataset.append('')
                #print(df_normal[t][i])
                print(dataset[i])
                continue
            train_y.append(int(df_normal['label(0=GE,1=NGGCT)'][i]))
            train_x.append(x_t)
            dataset.append('train')
        else:
            x_t=[]
            for t in key_list:
                if ''== df_normal[t][i] or 'NA' in str(df_normal[t][i]) or '无' in str(df_normal[t][i]):
                    #print(df_normal[t][i])
                    x_t.append(-1) 
                elif 'neg' in str(df_normal[t][i]).replace(' ','').lower() or '＜' in str(df_normal[t][i]).replace('<','＜'):
                    x_t.append(0)
                else:
                    try:
                        x_t.append(float(df_normal[t][i]))
                    except:
                        x_t.append(-1)

            if -1 in x_t or df_normal['label(0=GE,1=NGGCT)'][i]=='NA':
                dataset.append('')
                #print(df_normal[t][i])
                print(x_t,df_normal['PatientID'][i])
                continue
            if df_normal['StudyDate'][i]>20210000:
                test_y_2021.append(int(df_normal['label(0=GE,1=NGGCT)'][i]))
                test_x_2021.append(x_t)
                dataset.append('test_2021')
            else:
                test_y.append(int(df_normal['label(0=GE,1=NGGCT)'][i]))
                test_x.append(x_t)
                dataset.append('test')
    df_normal['dataset']=dataset
    return np.array(train_x).astype(np.float),train_y,np.array(test_x).astype(np.float),test_y,np.array(test_x_2021).astype(np.float),test_y_2021
def add_info(df,df2,add_list):
    new_dict={}
    for i in range(len(df2)):
        t_dict={k:df2[k][i] for k in df2.keys()}
        new_dict[df2['PatientID'][i]]=t_dict
    add_dict={t:[] for t in add_list}
    for i in range(len(df)):
        for k in add_list:
            add_dict[k].append(new_dict[df['PatientID'][i]][k])
    for k in add_list:
        df[k]=add_dict[k]
def age2num(l):
    if l.endswith('Y'):
        return int(l[:-1])
    if l.endswith('M'):
        return int(l[:-1])/12
    return l
def bingli2num(l):
    l=str(l).replace(',','').replace(' ','').replace('<','').replace('>','')  
    if 'neg' in l.lower():
        return 0
    else:
        return float(l)
def local2num(l):
    new_dict={
        '1':[],
        '2':[],
        '3':[],
        '4':[],
    }
    for t in l:
        t=str(t)
        for k in new_dict.keys():
            if k in t:
                new_dict[k].append(1)
            else:
                new_dict[k].append(0)
    return new_dict
def sex2num(l):
    if l=='M':
        return 0
    else:
        return 1
def perform2num(l):
    if l=='' or l=='NA':
        return 0
    else:
        return  int(l)
def get_matrix(gt,pre_logit,name='test',flag=0):
    gt=np.array(gt)
    conf_m=metrics.confusion_matrix(gt,pre_logit>0.5)
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
    return spe,sen,auc,acc,auc1,ppv,npv
def get_predict_result(train_x,train_y,test_x,test_y):
    mm=MinMaxScaler()
    train_x=mm.fit_transform(train_x)
    test_x=mm.transform(test_x)
    model = LogisticRegression(class_weight='balanced') #创建模型
    model.fit(train_x, train_y) #训练
    test_p=model.predict_proba(test_x)[:,1]
    train_p=model.predict_proba(train_x)[:,1]
    test_y=np.array(test_y)
    print(get_matrix(test_y,test_p))
    print(get_matrix(train_y,train_p))
    return test_p,test_y
    
def move_image(df,nii_path):
    mkdir(nii_path)
    for i in range(len(df)):
        ori_path=df['path'][i]
        if not os.path.exists(ori_path):
            continue
        try:
            shutil.copytree(ori_path,os.path.join(nii_path.replace('_nii','_dcm'),str(df['PatientID'][i])))
        except:
            continue
def change_t2_path(nii_path):
    for t in os.listdir(nii_path):
        tt=t.split('.')[0]
        mkdir(os.path.join(nii_path,tt))
        shutil.move(os.path.join(nii_path,t),os.path.join(nii_path,tt,'T2.nii.gz'))
def check_path(dir,xlsf,c_name):
    df=pd.read_excel(xlsf,keep_default_na=False)
    name_list=[t for t in os.listdir(dir)]
    is_list=[]
    for t in df['PatientID']:
        if str(t) in name_list:
            is_list.append(1)
        else:
            is_list.append(0)
    df[c_name]=is_list
    df.to_excel(xlsf.replace('.xls','_new.xls'),index=False)
def get_t2_list(dir):
    for t in os.listdir(dir):
        if len(os.listdir(os.path.join(dir,t)))<4:
            print(t)
def get_miss_patientID(p1,p2):
    name_list=[t for t in os.listdir(p1)]
    png_list=[t.split('.')[0] for t in os.listdir(p2)]
    for t in name_list:
        if not t in png_list:
            print(t)
def split_train_test_new(xlsf,p2):
    import random
    df1=pd.read_excel(xlsf,'pro',keep_default_na=False)
    png_list=[t.split('.')[0] for t in os.listdir(p2)]
    name_list=[]
    label_list=[]
    for i in range(len(df1)):
        if str(df1['PatientID'][i]) in png_list:
            name_list.append(df1['PatientID'][i])
            label_list.append(df1['病理分类(0=GE,1=NGGCT)'][i])
    df_all=pd.DataFrame({'name':name_list,'label':label_list})
    df_all.to_csv('./data/test_2021.csv',index=False)
def get_tumor_dict():
    tumor_dict={}
    dcm_path='/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/T2/'
    nii_path='/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/T2_extra/'
    for t in os.listdir(dcm_path):
        t_name=t.replace(' ','_')
        if os.path.exists(os.path.join(nii_path,t_name)):
            k=str(get_first_dcm(os.path.join(dcm_path,t))['PatientID'])
            tumor_dict[k]=os.path.join(nii_path,t_name,'mask.nii.gz')
    dcm_path='/hd/wjy/data_process/liyanong/image/9_18/'
    nii_path='/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/T2_extra/'
    for t in os.listdir(dcm_path):
        t_name=t.replace(' ','_')
        if os.path.exists(os.path.join(nii_path,t_name)):
            k=str(get_first_dcm(os.path.join(dcm_path,t))['PatientID'])
            tumor_dict[k]=os.path.join(nii_path,t_name,'mask.nii.gz')
    print(len(tumor_dict))
#get_all_dicom_info
    print(len(tumor_dict))
def move_image(path,save_path,test_f,mod_list=[]):
    mkdir(save_path)
    df=pd.read_csv(test_f)
    for t  in df['name']:
        for k,v in enumerate(mod_list):
            shutil.copy(os.path.join(path,t,v+'.nii.gz'),os.path.join(save_path,t+str(k).zfill(4)+'.nii.gz'))
def update_multi(test_f):
    mul_name=[t.split('.')[0] for t in os.listdir('image/0716_nii_jpg/')]
    df=pd.read_csv(test_f)
    drop_list=[i for i,v in enumerate(df['name']) if not str(v) in mul_name]
    df=df.drop(drop_list)
    df.to_csv(test_f.replace('.csv','_mult.csv'),index=False)
def df2dict(df_3000,key,add_flag=False):
    dict_3000={}
    for i in range(len(df_3000)):
        t_k=str(df_3000[key][i])
        if t_k=='':
            continue
        t_k=df_3000[key][i]
        t_dict={t:str(df_3000[t][i]) for t in df_3000.keys() if t is not key}
        if add_flag:
            t_dict['added_info']=''
        if t_k in dict_3000.keys():
            for t in t_dict.keys():
                if dict_3000[t_k][t]=='':
                    dict_3000[t_k][t]=t_dict[t]
                elif t_dict[t] in dict_3000[t_k][t]:
                    continue
                else:
                    dict_3000[t_k][t]+=t_dict[t]
        else:
            dict_3000[t_k]=t_dict
    return dict_3000
#################################################################
# cal_tuomr_volume
def get_max_area(gt):
    gt_label=skimage.measure.label(gt>0)
    if gt_label.max()==0:
        return 0
    max_area=0
    for i in range(gt_label.max()):
        i_area=(gt_label==i+1).sum()
        if i_area>max_area:
            max_area=i_area
    return max_area
def get_voxel_space(itk_img):
    space=itk_img.GetSpacing()
    r=1
    for t in space:
        r*=t
    return r
################################################################
# added info
def get_max_volume(nii_p):

    itk_img=sitk.ReadImage(nii_p)
    r=get_voxel_space(itk_img)
    #max_area=get_max_area(sitk.GetArrayFromImage(itk_img))
    max_area=(sitk.GetArrayFromImage(itk_img)).sum()
    return r*max_area
def add_volume(test_f):
    path='/hd/wjy/data_process/liyanong/image/0716_nii/'
    df=pd.read_csv(test_f)
    volume_list=[get_max_volume(os.path.join(path,t,'T2_mask.nii.gz')) for t in df['name']]
    df['volume']=volume_list
    df.to_csv(test_f,index=False)
def add_client_info(test_f,flag=0,dataset='dataset'):
    df=pd.read_csv(test_f)
    if flag==0:
        xlsf='data/8_12_huigu.xlsx'
        if os.path.exists(xlsf):
            info_df=pd.read_excel(xlsf,keep_default_na=False)
        else:
            info_df=pd.read_excel('./data/0716_info.xlsx','Sheet1',keep_default_na=False)
    else:
        xlsf='data/8_12_pro.xlsx'
        if os.path.exists(xlsf):
            info_df=pd.read_excel(xlsf,keep_default_na=False)
        else:
            info_df=pd.read_excel('./data/0716_info.xlsx','pro',keep_default_na=False)
    if not dataset in info_df.keys():
        info_df[dataset]=['' for i in range(len(info_df))]
    info_dict=df2dict(info_df,'PatientID')
    client_keys=[
        'PatientAge',
        'PatientSex',
        'local（鞍区=1，松果体区=2，基底节及丘脑=3，其他=4）',
        'bhcg',
        'AFP',
        'T1序列信号 （hyper=1，non-hyper=0）',
        'T2序列信号 （hypo=1，non-hypo=0）',
        '增强(轻度强化=1，中度强化=2，显著强化=3，无强化=0)',
        '强化均匀=1.强化不均=2',
        '强化方式（环形=1，结节状=2，斑片状=3，不规则形=4，线样强化）',
        '伴囊变坏死Y=1 N=0',
        '伴出血（有=1，无=0）',
        ]
    print(info_dict[df['name'][0]][client_keys[0]])
    new_dict=df.to_dict('list')
    new_dict['age']=[age2num(info_dict[t][client_keys[0]]) for t in df['name']]
    new_dict['sex']=[sex2num(info_dict[t][client_keys[1]]) for t in df['name']]
    local_dict=local2num([info_dict[t][client_keys[2]] for t in df['name']])
    for k,v in local_dict.items():
        new_dict['local='+k]=v
    new_dict[client_keys[3]]=[bingli2num(info_dict[t][client_keys[3]]) for t in df['name']]
    new_dict[client_keys[4]]=[bingli2num(info_dict[t][client_keys[4]]) for t in df['name']]
    new_dict['T1']=[perform2num(info_dict[t][client_keys[5]]) for t in df['name']]
    new_dict['T2']=[perform2num(info_dict[t][client_keys[6]]) for t in df['name']]
    new_dict['enhance']=[perform2num(info_dict[t][client_keys[7]]) for t in df['name']]
    new_dict['enhance_mean']=[perform2num(info_dict[t][client_keys[8]]) for t in df['name']]
    new_dict['enhance_type']=[perform2num(info_dict[t][client_keys[9]]) for t in df['name']]
    new_dict['cystolization']=[perform2num(info_dict[t][client_keys[10]]) for t in df['name']]
    new_dict['blood']=[perform2num(info_dict[t][client_keys[11]]) for t in df['name']]
    new_df=pd.DataFrame(new_dict)
    new_df.to_csv(test_f.replace('.csv','_client.csv'),index=False)
    test_name=os.path.split(test_f)[1].split('.')[0]
    info_df[dataset]=[test_name if t in list(df['name']) else info_dict[t]['dataset'] for t in info_df['PatientID']]
    info_df.to_excel(xlsf,index=False)
def gen_test_label(p,del_list=[]):
    name_list=[]
    label_list=[]
    save_p=os.path.join(p,'all_nii')
    mkdir(save_p)
    for t in os.listdir(p):
        if t in del_list:
            continue
        if 'GE' in t:
            for tt in os.listdir(os.path.join(p,t)):
                if tt.endswith('.nii.gz'):
                    shutil.copy(os.path.join(p,t,tt),os.path.join(save_p,tt.replace('.nii','_0000.nii')))
                    name_list.append(tt.split('.')[0])
                    label_list.append(0)
        if 'NG' in t:
            for tt in os.listdir(os.path.join(p,t)):
                if tt.endswith('.nii.gz'):
                    shutil.copy(os.path.join(p,t,tt),os.path.join(save_p,tt.replace('.nii','_0000.nii')))
                    name_list.append(tt.split('.')[0])
                    label_list.append(1)
    df=pd.DataFrame({'name':name_list,'label':label_list})
    df.to_csv(os.path.join(p,'label.csv'),index=False)
def add_predict_result(test_f,pre_f):
    test_df=pd.read_csv(test_f)
    pre_df=pd.read_csv(pre_f)
    pre_dict=df2dict(pre_df,'name')
    dl_list=[pre_dict[k]['all_voxel']  if  k in  pre_dict.keys() else '' for k in test_df['name']]
    test_df['dl_result']=dl_list
    test_df.to_csv(test_df,index=False)
#get_all_dicom_info
def first_step():
    s_dict=init_s_dict()
    path_list=[
        '/hd/wjy/data_process/liyanong/image/Gct_test_9_8/',
        '/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/GCT_all/',
        '/hd/wjy/data_process/liyanong/image/BBGCTS_BOLD_new/',
        '/hd/wjy/data_process/liyanong/image/9_18/T2_dcm',
    ]
    #get dicom path and base info
    for path in path_list:
    #遍歷每個病人
        for t in os.listdir(path):
                #遍歷每個序列
            #for tt in os.listdir(os.path.join(path,t)):
                if '.' in t:
                    continue
                try:
                    #update_study_info(os.path.join(path,t,tt),s_dict)
                    update_study_info(os.path.join(path,t),s_dict)
                except:
                    continue
    write_sdict(s_dict,'data/dicom_base_info.csv')
#add nii info
def second_step():
    nii_dict={
        '9_8':'/hd/wjy/data_process/liyanong/image/1202/nii',
        'GCTs':'/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/T2_nii',
        'BBGCTS_BOLD':'/hd/wjy/data_process/liyanong/image/BBGCTS_BOLD_T2/nii',
        '9_1':'/hd/wjy/data_process/liyanong/image/9_18/nii',
        '9_18':'/hd/wjy/data_process/liyanong/image/9_18/nii_2021',
    }
    #tumor_dict={
    #    '9_8':'/hd/wjy/data_process/liyanong/image/1202/all_test/',
    #    'GCTs':'/hd/wjy/data_process/liyanong/image/GCTs_Radiomics/T2_nii/',
    #    'BBGCTS_BOLD':'/hd/wjy/data_process/liyanong/image/BBGCTS_BOLD_T2/nii_pre/preTs_tumor',
    #    '9_1':'/hd/wjy/data_process/liyanong/image/9_18/nii_pre_tumor/',
    #    '9_18':'/hd/wjy/data_process/liyanong/image/9_18/',

    #}

    df=pd.read_csv('data/dicom_base_info.csv')
    name_dict={k:os.listdir(nii_dict[k]) for k in nii_dict.keys()}
    nii_paths=[]
    nii_name=[]
    for i in range(len(df)):
        for k in nii_dict.keys():
            if k in df['path'][i]:
                nii_path=nii_dict[k]
                nii_names=name_dict[k]
        flag=0
        if '1202' in nii_path:
            t_name=str(df['PatientID'][i])+str(df['StudyDate'][i])
            print(t_name)
            t_name=str(df['path'][i]).split('/')[-1].split('_')
            t_name=t_name[1]+t_name[0]
        if 'GCT' in nii_path:
            t_name=df['PatientName'][i].replace(' ','').replace('^','')
            #print(t_name)
        if '9_18' in nii_path:
            t_name=df['PatientName'][i].replace(' ','').replace('^','')
            #print(t_name)
        print(t_name,nii_path)
        for t in nii_names:
            if t.replace('_','').startswith(t_name):
                nii_paths.append(nii_path)
                nii_name.append(t[:-12])
                flag=1
                break
        if flag==0 and '9_1' in nii_path:
            nii_path=nii_dict['9_1']
            for t in name_dict['9_1']:
                if t.replace('_','').startswith(t_name):
                    nii_paths.append(nii_path)
                    nii_name.append(t[:-12])
                    flag=1
                    break
        if flag==0:
            nii_paths.append('')
            nii_name.append('')
    df['nii_path']=nii_paths
    df['nii_name']=nii_name
    df.to_csv('data/dicom_and_nii_info.csv')
# add label info
def third_step():
    df=pd.read_csv('data/dicom_and_nii_info.csv')
    new_dict=df.to_dict('list')
    new_dict['label_path']=['']*len(df)
    label_list=[
        '/hd/wjy/data_process/liyanong/data/train.csv',
        '/hd/wjy/data_process/liyanong/data/train_local.csv',
        '/hd/wjy/data_process/liyanong/data/train_new.csv',
        '/hd/wjy/data_process/liyanong/data/test_local.csv',
        '/hd/wjy/data_process/liyanong/data/test.csv',
        '/hd/wjy/data_process/liyanong/data/test_new.csv',
        '/hd/wjy/data_process/liyanong/data/label_2021.csv',
        '/hd/wjy/data_process/liyanong/data/label_extanded.csv',
        '/hd/wjy/data_process/liyanong/data/Gct_test_9_8.csv',
        '/hd/wjy/data_process/liyanong/data/Gct_test_9_8_new.csv',
    ]
    for l in label_list:
        add_label(new_dict,l)
    df=pd.DataFrame(new_dict)
    df.to_excel('data/dicom_and_nii_info_new.xlsx',index=False)
    # add client info
    df=pd.read_excel('data/副本dicom_and_nii_info_new_0619.xlsx','2022.06.18_18点',keep_default_na=False)
    new_dict=df.to_dict('list')
    df=pd.read_excel('data/常规判读_V2.xlsx',keep_default_na=False)
    add_client_info(new_dict,df)
    df=pd.DataFrame(new_dict)
    df.to_excel('data/dicom_and_nii_info_620.xlsx',index=False)
# add bingli
def four_step():
    #df=pd.read_excel('data/副本dicom_and_nii_info_623.xlsx','2022.06.25',keep_default_na=False)
    #new_dict=df.to_dict('list')
    #df=pd.read_excel('data/副本dicom_and_nii_info_623.xlsx','肿瘤标志物+病理',keep_default_na=False)
    #add_bingli_info(new_dict,df)
    #df=pd.DataFrame(new_dict)
    #df.to_excel('data/dicom_and_nii_info_629.xlsx',index=False)
    #df=pd.read_excel('data/dicom_and_nii_info_704.xlsx',' 修正 2022.7.04',keep_default_na=False)
    #new_dict=df.to_dict('list')
    #df=pd.read_excel('data/病理确定0704.xlsx',keep_default_na=False)
    #add_bingli_info(new_dict,df)
    #df=pd.DataFrame(new_dict)
    #df.to_excel('data/dicom_and_nii_info_704_new.xlsx',index=False)
    #df=pd.read_excel('data/副本DL_总表0709.xlsx','0712',keep_default_na=False)
    df=pd.read_excel('data/0716.xlsx','Sheet3',keep_default_na=False)
    new_dict=df.to_dict('list')
    ##df=pd.read_excel('data/病理确定0704.xlsx',keep_default_na=False)
    ##df=pd.read_excel('data/DL_总表0708.xlsx',keep_default_na=False)
    df=pd.read_excel('data/副本DL_总表0709.xlsx','Sheet3',keep_default_na=False)
    add_bingli_info(new_dict,df)
    df=pd.read_excel('data/dicom_and_nii_info_629.xlsx',keep_default_na=False)
    add_bingli_info(new_dict,df)
    df=pd.DataFrame(new_dict)
    df.to_excel('data/0716_info.xlsx',index=False)
    move_image(df,'image/0716_dcm_all')
    dir='image/0716_dcm_all'
    get_t2_list(dir)
    change_t2_path('image/0716_nii')
# preprocess data
def five_step():
    df=pd.read_excel('data/副本DL_总表0709.xlsx','0712',keep_default_na=False)
    df['PatientAge']=[age2num(t) for t in df['PatientAge']]
    new_dict=local2num(df['local（鞍区=1，松果体区=2，基底节及丘脑=3，其他=4）'])
    for k in new_dict:
        df['local='+k]=new_dict[k]
    bingli_list=['bhcg','AFP']
    for t in bingli_list:
        df[t]=[bingli2num(tt) for tt in df[t]]
    df.to_csv('bingli_0712.csv',index=False)
# split train and test
def six_step():
    #key_list=['bHCG血','AFP血','bHCG（CSF）','AFP血（CSF）']
    #df=pd.read_excel('data/DL_总表0706.xlsx',keep_default_na=False)
    #df2=pd.read_excel('data/dicom_and_nii_info_629.xlsx',keep_default_na=False)
    #add_info(df,df2,['label_path','nii_name'])
    #key_list=['bHCG血','AFP血','PatientAge','']
    ##key_list=['PatientAge']
    #df['PatientAge']=[age2num(t) for t in df['PatientAge']]
    #train_x,train_y,test_x,test_y,test_x_2021,test_y_2021=split_train_test(df,key_list)
    #get_predict_result(train_x,train_y,test_x,test_y)
    #df.to_excel('data/dicom_trian_and_nii_info_706.xlsx',index=False)
    #df=pd.read_excel('data/0716_info.xlsx',keep_default_na=False)
    #p1='image/0716_nii/'
    #get_miss_patientID(p1,p2)
    p2='image/0716_nii_png/'
    xlsf='data/0716_info.xlsx'
    split_train_test_new(xlsf,p2)
# dicom to nifty and dicom manage
def seven_step():
    from dicom2nifti.convert_dicom import dicom_series_to_nifti
    save_path='./image/0801_nii/'
    save_path_dcm='./image/0801_dcm/'
    ori_path='/hd/wjy/data_process/liyanong/0716_dcm_all/无T1C+T1/'
    name_list=[t for t in os.listdir('./image/0716_nii/')]
    for t in os.listdir(ori_path):
        if t in name_list:
            mkdir(os.path.join(save_path,t))
            mkdir(os.path.join(save_path_dcm,t))
            for tt in os.listdir(os.path.join(ori_path,t)):
                try:
                    if tt.endswith('T2'):
                        #dicom_series_to_nifti(os.path.join(ori_path,t,tt),os.path.join(save_path,t,'T2.nii.gz'))
                        shutil.copytree(os.path.join(ori_path,t,tt),os.path.join(save_path_dcm,t,'T2'))
                    if tt.endswith('T1'):
                        #dicom_series_to_nifti(os.path.join(ori_path,t,tt),os.path.join(save_path,t,'T1.nii.gz'))
                        shutil.copytree(os.path.join(ori_path,t,tt),os.path.join(save_path_dcm,t,'T1'))
                    if tt.endswith('T1C'):
                        #dicom_series_to_nifti(os.path.join(ori_path,t,tt),os.path.join(save_path,t,'T1C.nii.gz'))
                        shutil.copytree(os.path.join(ori_path,t,tt),os.path.join(save_path_dcm,t,'T1C'))
                except:
                    print(t,tt)
#manage all data to final version
def eight_step():
    train_f='./data/0725_train.csv'
    test_f='./data/0725_test.csv'
    pro_f='./data/test_2021.csv'
    #update_multi(train_f)
    #update_multi(test_f)
    #update_multi(pro_f)
    add_client_info(train_f)
    add_client_info(test_f)
    add_volume(test_f.replace('.csv','_client.csv'))
    add_volume(train_f.replace('.csv','_client.csv'))
    #add_client_info(pro_f,1)
    #add_client_info(train_f.replace('.csv','_mult.csv'))
    #add_client_info(test_f.replace('.csv','_mult.csv'))
    #add_client_info(pro_f.replace('.csv','_mult.csv'),1)
    #T1_paths=glob.glob('image/0801_dcm/*/T1/')
    #T2_paths=glob.glob('image/0801_dcm/*/T2/')
    #T1C_paths=glob.glob('image/0801_dcm/*/T1C/')
    #t1_dict=init_s_dict()
    #t2_dict=init_s_dict()
    #t1c_dict=init_s_dict()
    #for t in T1_paths:
    #    update_study_info(t,t1_dict)
    #for t in T2_paths:
    #    update_study_info(t,t2_dict)
    #for t in T1C_paths:
    #    update_study_info(t,t1c_dict)
    #write_sdict(t1_dict,'data/T1_dicom_info.csv')
    #write_sdict(t2_dict,'data/T2_dicom_info.csv')
    #write_sdict(t1c_dict,'data/T1C_dicom_info.csv')
# process huanhu and sanbo result
def night_step():
    #generate test.csv and copy T2 file
    path1='/hd/wjy/data_process/liyanong/image/1202/三博/'
    path2='/hd/wjy/data_process/liyanong/image/1202/环湖/'
    path11='/hd/wjy/data_process/liyanong/image/1202/all_test/result_0801/huanhu_test_T2_case.csv'
    path22='/hd/wjy/data_process/liyanong/image/1202/all_test/result_0801/sanbo_test_T2_case.csv'
    path1_label=os.path.join(path1,'label.csv')
    path2_label=os.path.join(path2,'label.csv')
    del_list=['qin_dan_yang','zhao_qing_xiang','cui_zhen_ming']
    gen_test_label(path1)
    gen_test_label(path2)
    # added client info
    client_xlsf='/hd/wjy/data_process/liyanong/data/sanbo+huanhu临床信息0816.xlsx'
    client_df_1=pd.read_excel(client_xlsf,'三博',keep_default_na=False)
    client_df_2=pd.read_excel(client_xlsf,'环湖',keep_default_na=False)
    key_dict={
        'age':'age',
        'sex':'gender（F=0，M=1）',
        'bhcg':'bhcg（F=0，M=1）',
        'AFP':'afp',
    }
    dict_1=df2dict(client_df_1,'name')
    dict_2=df2dict(client_df_2,'name')
    print(dict_1.keys())
    # added predict result and volume
# multi feature logistic regression t_test and kafang_test
def ten_step():
    test_f='data/0725_test_client.csv'
    train_f='data/0725_train_client.csv'
    save_f='data/0725_TTest.csv'
    key_list_T=[
        'age',
        'bhcg',
        'AFP',
        'volume',
        ]
    key_list_kafang=[
        'sex',
        'T1',
        'T2',
        'local=1',
        'local=2',
        'local=3',
        'local=4',
        'enhance',
        'enhance_mean',
        'enhance_type',
        'cystolization',
        'blood',
        ]
    get_all_result(train_f,test_f,key_list_T,save_f)
    get_kafa(train_f,test_f,key_list_kafang)
    key_list_feature=[
        'age',
        'bhcg',
        'AFP',
        'local=1',
        'local=2',
        'local=3',
        'enhance',
        'enhance_mean',
        ]
    train_x,train_y=read_feature(train_f,key_list_feature)
    test_x,test_y=read_feature(test_f,key_list_feature)
    #print(test_x)
    test_p,test_y=get_predict_result(train_x,train_y,test_x,test_y)
    #print(test_p)
    get_class_metrix(['Multivarite logistic regression of combined conventional information '],
            [test_y],
            [test_p],
            'data/result/multi_logistic_regression')
    test_t2_f='image/1202/all_test/result_0801/test_T2_case.csv'
    train_t2_f='image/1202/all_test/result_0801/train_T2_case.csv'
    test_multi_f='image/1202/all_test/result_0801/test_multi_case.csv'
    test_t2_df=pd.read_csv(test_t2_f)
    train_t2_df=pd.read_csv(train_t2_f)
    test_df=pd.read_csv(test_f)
    train_df=pd.read_csv(train_f)
    test_df['t2_predict']=test_t2_df['all-voxel']
    train_df['t2_predict']=train_t2_df['all-voxel']
    test_df.to_csv(test_f)
    train_df.to_csv(train_f)
    test_multi_df=pd.read_csv(test_multi_f)
    key_list_feature.append('t2_predict')
    train_x,train_y=read_feature(train_f,key_list_feature)
    test_x,test_y=read_feature(test_f,key_list_feature)
    #print(test_x)
    test_p_dl,test_y=get_predict_result(train_x,train_y,test_x,test_y)
    #print(test_p)
    get_class_metrix(['Multivarite logistic regression of combined conventional information ','Combination of conventional information and iGCT-T2 Net ','iGCT-T2 Net '],
            [test_y]*3,
            [test_p,test_p_dl,test_t2_df['all-voxel']],
            'data/result/multi_logistic_regression+t2Net')
    get_class_metrix(['iGCT-T2 Net ','MR multi-modalities (T1W,T2W and CE-T1W) Net '],
            [test_y,test_multi_df['label']],
            [test_t2_df['all-voxel'],test_multi_df['all-voxel']],
            'data/result/multiNet+t2Net')
    #print(get_matrix(test_p,test_y.astype(np.int16)))
# manage_addtion info
def get_pre_dict(csv_f):
    df=pd.read_csv(csv_f)
    new_dict={df['name'][i]:df['all-voxel'][i] for i in range(len(df))}
    return new_dict
def eleven_step():
    dev_xlsf = 'data/develop_set_GCT.xlsx'
    c1_xlsf='data/center1.xlsx'
    c2_xlsf='data/S_Table_Center2_1102.xlsx'
    c3_xlsf='data/S_Table_Center3_1102.xlsx'
    train_df=pd.read_excel(dev_xlsf,'training',keep_default_na=False)
    test_df=pd.read_excel(dev_xlsf,'test',keep_default_na=False)
    c1_df=pd.read_excel(c1_xlsf,'Sheet2',keep_default_na=False)
    c2_df=pd.read_excel(c2_xlsf,'Sheet2',keep_default_na=False)
    c3_df=pd.read_excel(c3_xlsf,'Sheet2',keep_default_na=False)
    train_y=[t for t in train_df['Pathological Diagnosis\n(0=GE,1=NGGCT)']]
    test_y=[t for t in test_df['Pathological Diagnosis\n(0=GE,1=NGGCT)']]
    c1_y=[t for t in c1_df['Pathological Diagnosis\n(0=GE,1=NGGCT)']]
    c2_y=[t for t in c2_df['Pathological Diagnosis\n(0=GE,1=NGGCT)']]
    c3_y=[t for t in c3_df['Pathological Diagnosis\n(0=GE,1=NGGCT)']]
    train_p=[t for t in train_df['Radiology Sign (GE=0,NGGCT=1)']]
    test_p=[t for t in test_df['Radiology Sign (GE=0,NGGCT=1)']]
    c1_p=[t for t in c1_df['Radiology Sign (GE=0,NGGCT=1)']]
    c2_p=[t for t in c2_df['Radiology Sign (GE=0,NGGCT=1)']]
    c3_p=[t for t in c3_df['Radiology Sign (GE=0,NGGCT=1)']]
    name_list=[
            'Develop set ',
            'Testing set ',
            'Internal validation 1 ',
            'External validation 2 ',
            'External validation 3 ',
    ]
    all_dict={
            'Develop set ':'image/1202/all_test/result_0801/train_T2_case.csv',
            'Testing set ':'image/1202/all_test/result_0801/test_T2_case.csv',
            'Internal validation 1 ':'image/1202/all_test/result_0801/pro_test_T2_case.csv',
            'External validation 2 ':'image/1202/all_test/result_0801/sanbo_test_T2_case.csv',
            'External validation 3 ':'image/1202/all_test/result_0801/huanhu_test_T2_case.csv',
            }
    all_pre=[get_pre_dict(all_dict[t]) for t in all_dict.keys()]
    c1_df['T2_predict']=[all_pre[2][t] for t in c1_df['PatientID']]
    c2_df['T2_predict']=[all_pre[3][t.replace(' ','_')] for t in c2_df['PatientID']]
    c3_df['T2_predict']=[all_pre[4][str(t).replace(' ','_')] if str(t).replace(' ','_') in all_pre[4].keys() else '' for t in c3_df['PatientID']]
    c1_df.to_csv('data/result/pro_info.csv',index=False)
    c2_df.to_csv('data/result/sanbo_info.csv',index=False)
    c3_df.to_csv('data/result/huanhu_info.csv',index=False)
    #get_class_metrix(name_list,
    #        [train_y,test_y,c1_y,c2_y,c3_y],
    #        [train_p,test_p,c1_p,c2_p,c3_p],
    #        'data/result/Radiology_Sign')
def twelve_step():
    #path='/Users/yanongli/Desktop/DL iGCT-Net 结果最终/result_1107/0725_test_client_1114.csv'
    #path='data/pro_info.csv'
    #path='data/sanbo_info.csv'
    path='data/huanhu_info.csv'
    df=pd.read_csv(path)
    name_list=[t for t in df.keys()[-6:]]
    get_class_metrix(name_list,
            [df['label']]*6,
            [df[t] for t in name_list],
            #'data/result/Radiology_pro')
            #'data/result/Radiology_sanbo')
            'data/result/Radiology_huanhu')

if __name__=='__main__':
    #seven_step()
    #get_tumor_dict()
    #six_step()
    #eight_step()
    #night_step()
    #ten_step()
    #eleven_step()
    twelve_step()