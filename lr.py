from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from cleaner import load_split
from metrics import report, mcnemar
from nb_alt import features, accuracy

def to_dicts(rows, groups):
    dicts=[]
    for headline, label in rows:
        d={}
        for f in features(headline, groups):
            d[f]=d.get(f, 0)+1
        dicts.append(d)
    return dicts

if __name__== "__main__":
    train_data=load_split("train.jsonl")
    dev_data=load_split("dev.jsonl")
    test_data=load_split("test.jsonl")
    train_gold=[]
    for headline, label in train_data:
        train_gold.append(label)
    dev_gold=[]
    for headline, label in dev_data:
        dev_gold.append(label)
    test_gold=[]
    for headline, label in test_data:
        test_gold.append(label)
    for name, groups in [("uni", ["uni"]), ("uni+bi", ["uni", "bi"])]:
        vec=DictVectorizer()
        x_train=vec.fit_transform(to_dicts(train_data, groups))
        x_dev=vec.transform(to_dicts(dev_data, groups))
        x_test=vec.transform(to_dicts(test_data, groups))
        best_c=None
        best_acc=-1
        for strength in [0.1, 0.3, 1, 3, 10]:
            model=LogisticRegression(C=strength, max_iter=2000)
            model.fit(x_train, train_gold)
            acc=accuracy(dev_gold, list(model.predict(x_dev)))
            if acc>best_acc:
                best_acc=acc
                best_c=strength
        model=LogisticRegression(C=best_c, max_iter=2000)
        model.fit(x_train, train_gold)
        test_pred=list(model.predict(x_test))
        report("lr "+name+" C "+str(best_c), test_gold, test_pred)
        nb_pred=[]
        with open("nb_alt_"+name+".txt") as f:
            for line in f:
                nb_pred.append(int(line))
        b, c, chi2, p=mcnemar(test_gold, test_pred, nb_pred)
        print("  mcnemar vs nb_alt: lr only right", b, "nb only right", c, "p", round(p, 4))
        with open("lr_"+name+".txt", "w") as f:
            for label in test_pred:
                f.write(str(label)+"\n")