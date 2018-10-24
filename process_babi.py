import sys
import re

# adds extra reasoning over latent space, compared to plain voting

def babiNN(filename):
    X_left=[]
    X_right=[]
    y=[]
    FeatDict={}
    Storage={}
    NN=[]
    evidence=[]
    
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    
    for i in range(0,len(lines)):
        m = re.match("^([^\#].+),([A-Z]),([A-Z])\s+\{",lines[i])
        if m:
            if len(Storage)!=0:
                j=1
                for nn in NN:
                    print j,' '.join(nn.split(","))+"."
                    j+=1
                output="%d %s? \t%s\t%s"%(j,' '.join(Storage["input"].split(",")),gt,' '.join(evidence))
                print output 
            fv=m.group(1)
            gt=m.group(2)
            pred=m.group(3)
            Storage["input"]=fv
            evidence=[]
            NN=[]
            evidence_str=""                    

            continue
        m=re.match("^\#\W(.+),\{\s+([A-Z]).+",lines[i])                    
        if m:
            fv_nn=m.group(1)
            gt_nn=m.group(2)
            features_nn=fv_nn.split(",")
            if fv != fv_nn:
                NN.append(fv_nn)
                if gt_nn==gt:
                    evidence.append(str(len(NN)))



if __name__=="__main__":
    babiNN(sys.argv[1])

