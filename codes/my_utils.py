import numpy as np
import random
from collections import defaultdict
from scipy.sparse import dok_matrix, lil_matrix

'''------------------------------new for TMM experiments------------------------------------------------------ '''

num_users, num_items = 0, 0


def get_pairwise_train_dataset(path='data/ml1m_train.dat'):
    global num_users, num_items
    print('loading pair-wise data from file %s ...' % path)
    user_input, item_i_input, item_j_input = [], [], []
    with open(path, 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            u, i, j = int(arr[0]), int(arr[1]), int(arr[2])
            user_input.append(u)
            item_i_input.append(i)
            item_j_input.append(j)
            if u > num_users:
                num_users = u
            if i > num_items or j > num_items:
                num_items = max(i, j)
    return num_users, num_items, user_input, item_i_input, item_j_input


def get_test_data(path='data/ml1m_test_ratings.lsvm'):
    global num_users, num_items
    print('loading test data from file %s ...' % path)
    testRatings = dict()
    testItems = dict()

    # better: read sampled data
    with open(path, 'r') as f:
        for u, line in enumerate(f):
            testRatings[u], testItems[u] = list(), list()
            for item_rating in line.strip().split(' '):
                if 'jester' in path:
                    item, rating = int(item_rating.split(':')[0]), float(item_rating.split(':')[1])
                else:
                    item, rating = int(item_rating.split(':')[0]), int(float(item_rating.split(':')[1]))
                testItems[u].append(item)
                testRatings[u].append(rating)

    return num_items, testItems, testRatings