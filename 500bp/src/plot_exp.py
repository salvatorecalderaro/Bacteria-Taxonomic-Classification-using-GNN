import yaml
import argparse
import matplotlib.pyplot as plt

dpi=1000
plt.rcParams["text.usetex"]=True



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k","--ksize", type=int,required=True, help="K-size")
    args = parser.parse_args()
    k= args.ksize 
    return k

def plot_accuracies(k):
    taxa=["PHYLUM","CLASS","ORDER","FAMILY","GENUS"]
    acc=[]
    f1=[]
    prec=[]
    rec=[]
    for t in taxa:
        path="../experiments/%s/results_%s_%d.yaml" %(t,t,k)
        with open(path,"rb") as file:
            data=yaml.safe_load(file)
            acc.append(data["Accuracy"]["Mean"])
            prec.append(data["Precision"]["Mean"])
            rec.append(data["Recall"]["Mean"])
            f1.append(data["F1-Score"]["Mean"])
    
    title="DBG (k=%d) + GNN" %(k)

    fig_path="../plots/dbg_gnn_%d_acc_f1.pdf" %(k)
    
    plt.figure()
    plt.title(title)
    plt.plot(taxa,acc,label="Accuracy",marker="s")
    #plt.plot(taxa,prec,label="Precision",marker="s")
    #plt.plot(taxa,rec,label="Recall",marker="s")
    plt.plot(taxa,f1,label="F1-Score",marker="s")
    plt.xlabel("Taxa")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()
    plt.cla()

    fig_path="../plots/dbg_gnn_%d_prec_rec.pdf" %(k)
    
    plt.figure()
    plt.title(title)
    plt.plot(taxa,prec,label="Precision",marker="s")
    plt.plot(taxa,rec,label="Recall",marker="s")
    plt.xlabel("Taxa")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()
    plt.cla()


def main():
    k=parse_arguments()
    plot_accuracies(k)

if __name__=="__main__":
    main()