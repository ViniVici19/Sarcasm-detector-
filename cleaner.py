import json
import re
import random
DATA_PATH = "Sarcasm_Headlines_Dataset.json"
SEED = 42
def headlines(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                item=json.loads(line)
                rows.append((item["headline"].strip(), int(item["is_sarcastic"])))
    return rows

def rem_dupes(rows):
    labels={}
    for headline, label in rows:
        if headline not in labels:
            labels[headline]=set()
        labels[headline].add(label)
    clean=[]
    conflicts=0
    for headline in labels:
        if len(labels[headline])==1:
            clean.append((headline, list(labels[headline])[0]))
        else:
            conflicts +=1
    return clean, conflicts
def split(rows, seed):
    random.seed(seed)
    train = []
    dev= []
    test= []

    for label in [0, 1]:
        group =[]
        for row in rows:
            if row[1] ==label:
                group.append(row)
        random.shuffle(group)
        a = int(0.8*len(group))
        b= int(0.9*len(group))
        train+=group[:a]
        dev+=group[a:b]
        test+=group[b:]

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test

def tokenize(headline):
    headline=headline.lower().replace("\u200b", "")
    return re.findall(r"\w+(?:'\w+)?|[^\s\w]", headline)

def save_split(rows, path):
    with open (path, "w", encoding="utf-8" ) as f:
        for headline, label in rows:
            f.write(json.dumps({"headline": headline, "label": label})+"\n")

def load_split(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            item=json.loads(line)
            rows.append((item["headline"], item["label"]))
    return rows

if __name__=="__main__":
    rows=headlines(DATA_PATH)
    rows, conflicts = rem_dupes(rows)
    train, dev, test = split(rows, SEED)
    save_split(train, "train.jsonl")
    save_split(dev, "dev.jsonl")
    save_split(test, "test.jsonl")
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        sarcastic=0
        for headline, label in part:
            sarcastic+=label
        print(name, len(part), "sarcastic", sarcastic)
    print(tokenize(train[0][0]))