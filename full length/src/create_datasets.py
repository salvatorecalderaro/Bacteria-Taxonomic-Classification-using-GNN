from Bio import SeqIO
import pandas as pd
import numpy as np
from random import choice
from tqdm import tqdm
import h5py

def process_sequence(seq):  
    seq = seq.replace("N", choice(["A","C","G","T"]))
    seq = seq.replace("R", choice(["A","G"]))
    seq = seq.replace("S", choice(["G","C"]))
    seq = seq.replace("Y", choice(["C","T"]))
    seq = seq.replace("K", choice(["G","T"]))
    seq = seq.replace("M", choice(["A","C"]))
    seq = seq.replace("W", choice(["A","T"]))
    seq = seq.replace("B", choice(["C","G","T"]))
    seq = seq.replace("D", choice(["A","G","T"]))
    seq = seq.replace("H", choice(["A","C","T"]))
    seq = seq.replace("V", choice(["A","C","G"]))
    return seq

def create_datasets():
    """Create all the datasets.
    """
    sequences=[]
    ids=[]
    lenghts=[]
    path="../raw data/16S.fas"

    taxonomy_path="../raw data/taxonomy.csv"

    taxonomy=pd.read_csv(taxonomy_path)

    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence=process_sequence(str(record.seq).upper())
            sequences.append(sequence)
            lenghts.append(len(sequence))
            ids.append(record.name)
        
    taxa=["PHYLUM","CLASS","ORDER","FAMILY","GENUS"]
    for t in taxa:
        print("Create %s dataset...." %(t))
        labels_names=[]
        for id_seq in tqdm(ids):
            x=taxonomy.loc[taxonomy['Sequence'] == id_seq]
            l=x[t].values[0]
            labels_names.append(l)
        
        class_names=list(np.unique(labels_names))
        print("Number of sequences in the file:",len(sequences)) 
        #print("Classes:",class_names)
        print("Avg. Length %s SD %s"  %(str(np.mean(lenghts)),str(np.std(lenghts))))
        print("Number of classes:",len(class_names))

        mapping_labels=dict(list(zip(class_names,list(range(0,len(class_names))))))
        labels=list(map(mapping_labels.get, labels_names))
        dt = h5py.special_dtype(vlen=str) 

        path="../datasets/16S_%s.h5" %(t)
        with h5py.File(path,"w") as file:
            file.create_dataset('sequences', data=sequences,compression="gzip", compression_opts=9)
            file.create_dataset('labels', data=labels,compression="gzip", compression_opts=9)
            file.create_dataset('classes', data=class_names,compression="gzip", compression_opts=9,dtype=dt)

def main():
    """Main routine
    """
    print("Create 16S datasets.....")
    create_datasets()
    print("End!")
    exit(0)


if __name__=="__main__":
    main()