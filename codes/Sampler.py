import numpy as np
import random
from scipy.sparse import lil_matrix
from multiprocessing import Process, Queue


def sample_fun(user,
               item_i,
               item_j,
               batch_size,
               result_queue):

    while True:
        randnum = random.randint(0,100)
        print("Fix the random seed as: ", randnum)
        random.seed(randnum)
        random.shuffle(user)
        random.seed(randnum)
        random.shuffle(item_i)
        random.seed(randnum)
        random.shuffle(item_j)

        int_num_batch = int(len(user) / batch_size) 
 
        for i in range(int_num_batch):
            # deal with one batch
            batch_user = user[i * batch_size:(i + 1) * batch_size]
            batch_item_i = item_i[i * batch_size:(i + 1) * batch_size]
            batch_item_j = item_j[i * batch_size:(i + 1) * batch_size]

            result_queue.put((batch_user, batch_item_i, batch_item_j))
        


            result_queue.put((batch_user, batch_item_i, batch_item_j))



class Sampler(object):

    def __init__(self,
                 user,
                 item_i,
                 item_j,
                 batch_size,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []

        for i in range(n_workers):
            print("Workers %d begin working!" % i)
            self.processors.append(
                Process(target=sample_fun, args=(user, item_i, item_j, batch_size, self.result_queue)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for process in self.processors:  # Process
            process.terminate()
            process.join()
