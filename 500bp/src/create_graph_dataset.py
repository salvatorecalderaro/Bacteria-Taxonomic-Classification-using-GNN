import argparse
import h5py
from time import perf_counter as pc
from datetime import timedelta
from seq2graph import create_graph
import torch
import numpy as np
from tqdm import tqdm
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--taxa", type=str,required=True, choices=["PHYLUM","CLASS","ORDER","FAMILY","GENUS"], help="Taxa")
    parser.add_argument("-k","--ksize", type=int,required=True, help="K-size")
    args = parser.parse_args()
    c= args.taxa
    k=args.ksize
    return c,k


def load_data(c):
    path="../datasets/16S_%s_500bp.h5" %(c)
    with h5py.File(path, 'r') as file:
        sequences=list(file.get("sequences"))
        labels=list(file.get('labels'))



    print("%s:" %(c))
    print("Number of sequences in the dataset:",len(sequences))
    print("Number of classes:", len(np.unique(labels)))
    return sequences,labels


def create_graphs(sequences,labels,c,k):
    graphs=[]
    print("Building graphs...")
    start=pc()
    for s,l in tqdm(zip(sequences,labels),total=len(sequences)):
        seq=s.decode('utf8')   
        g=create_graph(seq,k)
        g.y=torch.tensor(l)
        graphs.append(g)
    end=pc()
    t=end-start
    total_time=timedelta(seconds=t)
    print("Ended in:" ,str(total_time))

    path="../experiments/%s/16S_%s_%d" %(c,c,k)

    with open(path,"wb") as file:
        pickle.dump(graphs,file)

    path="../experiments/%s/time_16S_%s_%d.txt" %(c,c,k)

    with open(path, 'w') as file:
        file.write("%s\n" %(c))
        file.write("Number of sequences: %d \n" %(len(graphs)))
        file.write("Total time: %s" %(str(t)))
    
def main():
    c,k=parse_arguments()
    sequences,labels=load_data(c)
    create_graphs(sequences,labels,c,k)
    exit(0)

if __name__=="__main__":
    main()