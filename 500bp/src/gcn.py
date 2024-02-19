import os 
import random
import torch
import argparse
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from model import GCN,train_net,predict
from torchmetrics.classification import Accuracy,Precision,Recall,F1Score,MatthewsCorrCoef
import yaml
import pandas as pd
import platform
import cpuinfo


seed=0
nfolds=10
mini_batch_size=32
hidden_channels=128
lr=0.001

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)



def identify_device():
    so=platform.system()
    if (so=="Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name=cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d=str(device)
        if d=='cuda':
            dev_name=torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name=cpuinfo.get_cpu_info()["brand_raw"]
    return device,dev_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--taxa", type=str,required=True, choices=["PHYLUM","CLASS","ORDER","FAMILY","GENUS"], help="Taxa")
    parser.add_argument("-k","--ksize", type=int,required=True, help="K-size")
    parser.add_argument("-e","--epochs", type=int,required=True, help="Number of epochs")
    args = parser.parse_args()
    c= args.taxa
    k=args.ksize
    e=args.epochs
    return c,k,e

def load_data(c,k):
    labels=[]
    path="../experiments/%s/16S_%s_%d" %(c,c,k)
    with open(path,"rb") as file:
        sequences=pickle.load(file)
    
    for s in sequences:
        labels.append(int(s.y))
    return sequences,labels

def create_train_test_loader(sequences,train,test):
    data=[]
    for i in train:
        data.append(sequences[i])
    print("Number of sequences in the training set:",len(data))
    trainloader = DataLoader(data, batch_size=mini_batch_size, shuffle=True)

    data=[]
    for i in test:
        data.append(sequences[i])
    print("Number of sequences in the test set:",len(data))
    testloader = DataLoader(data, batch_size=mini_batch_size, shuffle=False)

    return trainloader,testloader


def compute_metrics(y_true,y_pred,nclasses,f):
    accuracy=Accuracy(task="multiclass",num_classes=nclasses)
    acc=accuracy(y_true,y_pred).item()
    
    precision=Precision(task="multiclass",num_classes=nclasses,average="weighted")
    prec=precision(y_true,y_pred).item()

    recall=Recall(task="multiclass",num_classes=nclasses,average="weighted")
    rec=recall(y_true,y_pred).item()

    f1score=F1Score(task="multiclass",num_classes=nclasses,average="weighted")
    f1=f1score(y_true,y_pred).item()

    mcc_coeff=MatthewsCorrCoef(task="multiclass",num_classes=nclasses)
    mcc=mcc_coeff(y_true,y_pred).item()

    print("Accuracy",acc)
    print("Precision",prec)
    print("Recall",rec)
    print("F1-Score",f1)
    print("MCC",mcc)

    return [f,acc,prec,rec,f1,mcc]

def save_metrics(metrics,c,k):
    columns=["Fold","Accuracy","Precision","Recall","F1-Score","MCC"]
    data=pd.DataFrame(metrics,columns=columns)
    path="../experiments/%s/metrics_%s_%d.csv" %(c,c,k)
    data.to_csv(path,index=False)

    avg_metrics=[]
    yaml_data={}

    acc=data['Accuracy'].values
    avg=np.mean(acc)
    sd=np.std(acc)
    avg_metrics.append(("Accuracy",avg,sd))
    yaml_data["Accuracy"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    prec=data['Precision'].values
    avg=np.mean(prec)
    sd=np.std(prec)
    avg_metrics.append(("Precision",avg,sd))
    yaml_data["Precision"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    rec=data['Recall'].values
    avg=np.mean(rec)
    sd=np.std(rec)
    avg_metrics.append(("Recall",avg,sd))
    yaml_data["Recall"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    f1score=data['F1-Score'].values
    avg=np.mean(f1score)
    sd=np.std(f1score)
    avg_metrics.append(("F1-Score",avg,sd))
    yaml_data["F1-Score"]={"Mean":float(avg),"Standard Deviation":float(sd)}


    mcc=data['MCC'].values
    avg=np.mean(mcc)
    sd=np.std(mcc)
    avg_metrics.append(("MCC",avg,sd))
    yaml_data["MCC"]={"Mean":float(avg),"Standard Deviation":float(sd)}

    path="../experiments/%s/results_%s_%d.yaml" %(c,c,k)

    with open(path,"w") as file:
        yaml.dump(yaml_data,file)
    
    for m in avg_metrics:
        print("AVG. %s = %f SD = %f \n" %(m[0],m[1],m[2]))


def save_report(times,c,k,epochs,devname):

    path="../experiments/%s/times_%s_%d.csv" %(c,c,k)
    columns=["Fold","Training time","Testing time"]

    df=pd.DataFrame(times,columns=columns)
    df.to_csv(path,index=False)

    train_time_mu=np.mean(df["Training time"].values)
    train_time_sigma=np.std(df["Training time"].values)
    
    test_time_mu=np.mean(df["Testing time"].values)
    test_time_sigma=np.std(df["Testing time"].values)
    
    yaml_data={}
    yaml_data["Device"]=devname
    yaml_data["Training Time"]={
        "Mean":float(train_time_mu),
        "SD":float(train_time_sigma)
    }
    
    yaml_data["Testing Time"]={
        "Mean":float(test_time_mu),
        "SD":float(test_time_sigma)
    }

    yaml_data["Epochs"]=int(epochs)

    path="../experiments/%s/times_avg_%s_%d.yaml" %(c,c,k)

    with open(path,"w") as file:
        yaml.dump(yaml_data,file)

def main():
    device,devname=identify_device()
    print("Using %s - %s" %(device,devname))
    times=[]
    metrics=[]
    c,k,epochs=parse_arguments()
    sequences,labels=load_data(c,k)
    nclasses=len(np.unique(labels))
    skf=StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=seed)
    print("16S - %s, DBG (k=%d) + GNN (%d epochs)" %(c,k,epochs))
    nclasses=len(np.unique(labels))
    in_features=(k-1)*4
    for f,(train,test) in enumerate(skf.split(sequences,labels)):
        print("================================================")
        print("FOLD",f+1)
        trainloader,testloader=create_train_test_loader(sequences,train,test)
        net=GCN(in_features,hidden_channels,nclasses)
        net,train_time=train_net(device,net,trainloader,epochs,lr)
        y_true,y_pred,test_time=predict(device,net,testloader)
        m=compute_metrics(y_true,y_pred,nclasses,f+1)
        metrics.append(m)
        times.append((f+1,train_time,test_time))
        print("================================================")

    save_metrics(metrics,c,k)
    save_report(times,c,k,epochs,devname)
    exit(0)

if __name__=="__main__":
    main()