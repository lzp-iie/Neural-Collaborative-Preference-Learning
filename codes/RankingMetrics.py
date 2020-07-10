import numpy as np
import math
import heapq


_testItems = None
_testRatings = None
_K = None
_item_rating_dict = None

def evaluate_model(self, epoch):
    global _testItems
    global _testRatings
    global _K
    _testItems = self.testItems
    _testRatings = self.testRatings
    _K = self.topK
    _self = self

    metrics = np.array([0. for _ in range(6)])

    for user in _testItems.keys():
            metrics += eval_one_user(_self, user, epoch)

    return metrics / len(_testItems)


def eval_one_user(self, user, epoch):
    global _item_rating_dict
    ratings = _testRatings[user]
    items = _testItems[user]
    item_rating_dict = dict()

    for item, rating in zip(items, ratings):
        item_rating_dict[item] = rating
    
    _item_rating_dict = item_rating_dict

    k_largest_items = heapq.nlargest(_K, item_rating_dict, key=item_rating_dict.get)
    real_largest_items = heapq.nlargest(len(item_rating_dict), item_rating_dict, key=item_rating_dict.get)

    with open('output_%s/rating_NCPL_%s_e_%d.txt' % (self.dataset, self.dataset, epoch), 'a+') as out_rating_file:
        print(' '.join([str(_) for _ in real_largest_items]), file=out_rating_file)
    
    users = np.full(len(items), user, dtype='int32').tolist()
    predictions = self.predict(users, items)

    item_prediction_dict = dict()
    with open('output_%s/predict_ratings_NCPL_%s_e_%d.txt' % (self.dataset, self.dataset, epoch), 'a+') as out_p_r:
        for item, prediction in zip(items, predictions):
            item_prediction_dict[item] = prediction
            out_p_r.write("%s:%s " % (str(item), str(float(prediction))))
        out_p_r.write('\n')

    sorted_item = heapq.nlargest(len(item_rating_dict), item_prediction_dict, key=item_prediction_dict.get)
    
    with open('output_%s/prediction_NCPL_%s_e_%d.txt' % (self.dataset, self.dataset, epoch), 'a+') as out_prediction_file:
        print(' '.join([str(_) for _ in sorted_item]), file=out_prediction_file)
    
    top_labels = [1 if item in k_largest_items else 0 for item in sorted_item]

    hr = getHitRatio(top_labels[:_K])
    p = getPrecision(top_labels[:_K])
    ndcg_bin = getNDCG_bin(top_labels[:_K])
    auc = getAUC(top_labels)
    map = getMAP(top_labels)
    mrr = getMRR(top_labels)
    return np.array([hr, p, ndcg_bin, auc, map, mrr])


def getHitRatio(labels):
    return 1 if 1 in labels else 0


def getPrecision(labels):
    return sum(labels) / len(labels)

def getRecall(labels):
    pass

def getNDCG_bin(labels):
    dcg, max_dcg = 0, 0
    for i, label in enumerate(labels):
        dcg += label / math.log2(i + 2)
        max_dcg += 1 / math.log2(i + 2)
    return dcg / max_dcg


def getAUC(labels):
    global _K
    if len(labels) <= _K:
        return 1

    auc = 0
    for i, label in enumerate(labels[::-1]):
        auc += label * (i + 1)

    return (auc - _K * (_K + 1) / 2) / (_K * (len(labels) - _K))


def getMAP(labels):
    global _K
    MAP = 0
    for i, label in enumerate(labels):
        MAP += label * getPrecision(labels[:i + 1])
    return MAP / _K


def getMRR(labels):
    global _K
    mrr = 0
    for i, label in enumerate(labels):
        mrr += label * (1 / (i + 1))
    return mrr / _K