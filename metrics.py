import math

def confusion_matrix(gold, pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

def scores(gold, pred):
    tp, fp, fn, tn = confusion_matrix(gold, pred)
    accuracy = (tp + tn) / len(gold)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return accuracy, precision, recall, f1

def report(name, gold, pred):
    tp, fp, fn, tn = confusion_matrix(gold, pred)
    accuracy, precision, recall, f1 = scores(gold, pred)
    print(name)
    print("  accuracy  %.4f" % accuracy)
    print("  precision %.4f" % precision)
    print("  recall    %.4f" % recall)
    print("  f1        %.4f" % f1)
    print("                 pred 1  pred 0")
    print("  gold 1 (sarc)  %6d  %6d" % (tp, fn))
    print("  gold 0 (not)   %6d  %6d" % (fp, tn))

def mcnemar(gold, pred_a, pred_b):
    b = 0
    c = 0
    for g, a, p in zip(gold, pred_a, pred_b):
        if a == g and p != g:
            b += 1
        elif a != g and p == g:
            c += 1
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return b, c, chi2, p_value