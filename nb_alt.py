import math
from cleaner import tokenize, load_split
from metrics import report, mcnemar

def features(headline, groups):
    words=tokenize(headline)
    feats=[]
    if "uni" in groups:
        for word in words:
            feats.append("uni:"+word)
    if "bi" in groups:
        for i in range(1, len(words)):
            feats.append("bi:"+words[i-1]+"_"+words[i])
    return feats

def train(rows, groups):
    counts={0:{}, 1:{}}
    totals={0:0, 1:0}
    docs={0:0, 1:0}
    vocab=set()
    for headline, label in rows:
        docs[label]+=1
        for f in features(headline, groups):
            counts[label][f]=counts[label].get(f, 0)+1
            totals[label]+=1
            vocab.add(f)
    model={}
    model["counts"]=counts
    model["totals"]=totals
    model["vocab"]=vocab
    model["prior"]={0:docs[0]/len(rows), 1:docs[1]/len(rows)}
    return model

def score(model, feats, label, alpha):
    total=math.log(model["prior"][label])
    denom=model["totals"][label]+alpha*len(model["vocab"])
    for f in feats:
        if f in model["vocab"]:
            total+=math.log((model["counts"][label].get(f, 0)+alpha)/denom)
    return total

def classify(model, headline, groups, alpha):
    feats=features(headline, groups)
    if score(model, feats, 1, alpha)>score(model, feats, 0, alpha):
        return 1
    return 0

def predict(model, rows, groups, alpha):
    pred=[]
    for headline, label in rows:
        pred.append(classify(model, headline, groups, alpha))
    return pred

def accuracy(gold, pred):
    correct=0
    for g, p in zip(gold, pred):
        if g==p:
            correct+=1
    return correct/len(gold)

if __name__=="__main__":
    train_data=load_split("train.jsonl")
    dev_data=load_split("dev.jsonl")
    test_data=load_split("test.jsonl")
    dev_gold=[]
    for headline, label in dev_data:
        dev_gold.append(label)
    test_gold=[]
    for headline, label in test_data:
        test_gold.append(label)
    nb_pred=[]
    with open("naive_bayes.txt") as f:
        for line in f:
            nb_pred.append(int(line))
    for name, groups in [("uni", ["uni"]), ("uni+bi", ["uni", "bi"])]:
        model=train(train_data, groups)
        best_alpha=None
        best_acc=-1
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.80, 0.85, 0.9, 1.0]:
            acc=accuracy(dev_gold, predict(model, dev_data, groups, alpha))
            if acc>best_acc:
                best_acc=acc
                best_alpha=alpha
        test_pred=predict(model, test_data, groups, best_alpha)
        report("nb_alt "+name+" alpha "+str(best_alpha), test_gold, test_pred)
        b, c, chi2, p=mcnemar(test_gold, test_pred, nb_pred)
        print("  mcnemar vs nb: nb_alt only right", b, "nb only right", c, "p", round(p, 4))
        with open("nb_alt_"+name+".txt", "w") as f:
            for label in test_pred:
                f.write(str(label)+"\n")