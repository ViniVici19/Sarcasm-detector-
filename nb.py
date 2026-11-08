import math
from cleaner import tokenize, load_split
from metrics import report

def add(table, key, amount):
    table[key]=amount+ table.get(key, 0)

def build_vocab(rows):
    vocab=set()
    for headline, label in rows:
        for word in tokenize(headline):
            vocab.add(word)
    vocab.add("<unk>")
    vocab.add("</s>")
    return vocab

def prep(headline, vocab):
    words=["<s>"]
    for word in tokenize(headline):
        if word in vocab:
            words.append(word)
        else:
            words.append("<unk>")
    words.append("</s>")
    return words

def train( rows, label, vocab):
    bigram={}
    for headline, gold in rows:
        if gold!=label:
            continue
        words=prep(headline, vocab)
        for i in range(1, len(words)):
            add(bigram, (words[i-1], words[i]), 1)
    history={}
    followers={}
    left={}
    for pair in bigram:
        add(history, pair[0], bigram[pair])
        add(followers, pair[0], 1)
        add(left, pair[1], 1)
    model = {}
    model["bigram"] = bigram
    model["history"] = history
    model["followers"] = followers
    model["left"] = left
    model["bigram_types"] = len(bigram)
    model["continuation_types"] = len(left)
    return model
 
 
def prob(model, v, w, d, vocab_size):
    total = model["bigram_types"]
    unigram = max(model["left"].get(w, 0) - d, 0) / total
    unigram += d * model["continuation_types"] / total / vocab_size
    count_v = model["history"].get(v, 0)
    if count_v == 0:
        return unigram
    bigram = max(model["bigram"].get((v, w), 0) - d, 0) / count_v
    weight = d * model["followers"][v] / count_v
    return bigram + weight * unigram
 
 
def log_prob(model, words, d, vocab_size):
    total = 0.0
    for i in range(1, len(words)):
        total += math.log(prob(model, words[i - 1], words[i], d, vocab_size))
    return total

def classify(models, priors, headline, vocab, d):
    words = prep(headline, vocab)
    score0 = math.log(priors[0]) + log_prob(models[0], words, d, len(vocab))
    score1 = math.log(priors[1]) + log_prob(models[1], words, d, len(vocab))
    if score1 > score0:
        return 1
    return 0
 
 
def predict(models, priors, rows, vocab, d):
    pred = []
 
    for headline, label in rows:
        pred.append(classify(models, priors, headline, vocab, d))
    return pred
 
 
if __name__ == "__main__":
    train_data = load_split("train.jsonl")
    dev_data = load_split("dev.jsonl")
    test_data = load_split("test.jsonl")
    vocab = build_vocab(train_data)
    models = {0: train(train_data, 0, vocab), 1: train(train_data, 1, vocab)}
    sarcastic = 0
    for headline, label in train_data:
        sarcastic += label
    priors = {1: sarcastic / len(train_data), 0: 1 - sarcastic / len(train_data)}
    dev_gold = []
    for headline, label in dev_data:
        dev_gold.append(label)
    best_d = None
    best_acc = -1
    for d in [0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 1.00]:
        pred = predict(models, priors, dev_data, vocab, d)
        correct = 0
        for g, p in zip(dev_gold, pred):
            if g == p:
                correct += 1
        if correct > best_acc:
            best_acc = correct
            best_d = d
    print("best d", best_d)
    test_gold = []
    for headline, label in test_data:
        test_gold.append(label)
    test_pred = predict(models, priors, test_data, vocab, best_d)
    report("naive bayes, interpolated kneser-ney bigrams", test_gold, test_pred)
    with open("naive_bayes.txt", "w") as f:
        for p in test_pred:
            f.write(str(p) + "\n")

