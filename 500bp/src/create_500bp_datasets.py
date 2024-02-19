from tqdm import tqdm
import h5py
import numpy as np
import random

def get_random_subSeq(fullSeq, subSeq_len):
    idx = random.randrange(0, len(fullSeq) - subSeq_len + 1)
    
    return fullSeq[idx : (idx + subSeq_len)]

def create_dataset(c):
    path="../../full length/datasets/16S_%s.h5" %(c)
    seqs_500_bp=[]
    with h5py.File(path, 'r') as file:
        sequences=list(file.get("sequences"))
        labels=list(file.get('labels'))
        classes=list(file.get("classes"))

    for s in tqdm(sequences):
        seq=get_random_subSeq(s,500)
        seqs_500_bp.append(seq)
    
    dt = h5py.special_dtype(vlen=str) 
    path="../datasets/16S_%s_500bp.h5" %(c)
    with h5py.File(path,"w") as file:
        file.create_dataset('sequences', data=seqs_500_bp,compression="gzip", compression_opts=9)
        file.create_dataset('labels', data=labels,compression="gzip", compression_opts=9)
        file.create_dataset('classes', data=classes,compression="gzip", compression_opts=9,dtype=dt)
    
    

    print("%s:" %(c))
    print("Number of sequences in the dataset:",len(sequences))
    print("Number of classes:", len(np.unique(labels)))
    return sequences,labels

def main():
    taxa=["PHYLUM","CLASS","ORDER","FAMILY","GENUS"]
    print("Create 16S datasets (500 bp) .....")
    for t in taxa:
        create_dataset(t)
    print("End!")
    exit(0)


if __name__=="__main__":
    main()