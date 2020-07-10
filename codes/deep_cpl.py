import tensorflow as tf
import time
import numpy as np
import random

from RankingMetrics import evaluate_model
from Sampler import Sampler
from time import time


class Deep_CPL(object):
    def __init__(self,
                 sess,
                 dataset,
                 num_user,
                 num_item,
                 learning_rate=0.005,
                 reg_rate=0.1,
                 margin=1.9,
                 epoch=50,
                 batch_size=256,
                 verbose=True,
                 t=1,
                 display_step=100,
                 num_factor=10,
                 num_factor_mlp=64,
                 hidden_dimension=10,
                 topK=10,
                 lr_decay_rate=0.96,
                 ranker_layers=[128, 64, 32, 8],
                 test_n=70):

        self.sess = sess
        self.dataset = dataset
        self.num_user = num_user
        self.num_item = num_item

        self.starter_lr = learning_rate

        self.reg_rate = reg_rate

        self.margin = margin
        self.epochs = epoch
        self.usual_batch_size = batch_size
        self.last_batch_size = None
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step
        self.num_factor = num_factor
        self.num_factor_mlp = num_factor_mlp
        self.hidden_dimension = hidden_dimension
        self.topK = topK
        self.lr_decay_rate = lr_decay_rate
        self.ranker_layers = ranker_layers
        self.test_n = test_n

        
        self.P = None
        self.Q = None
        self.mlp_P = None
        self.mlp_Q = None

        self.learning_rate = None

        self.test_data = None
        self.user = None
        self.item_i = None
        self.item_j = None
        self.label = None

        self.testItems = None
        self.testRatings = None

        self.num_training = None
        self.total_batch = None

        self.pred_pos_y = None
        self.pred_neg_y = None
        self.pred_y = None
        self.ranking_loss = None
        self.loss = None
        self.optimizer = None
        self.sampler = None
        self.validation_users = None
        self.epochs_Loss = None
        self.epochs_reg_Loss = None
        self.one_batch_user_list = None
        self.one_batch_pos_neg_list = None
        self.pred_befor_sigmoid = None


        print("Deep_CPL model has been initialized! ")

    def build_network(self):
        self.one_batch_user_list = tf.placeholder(tf.int32, [None],
                                                  name="one_batch_user_list")
        self.one_batch_pos_neg_list = tf.placeholder(
            tf.int32, [None], name="one_batch_pos_neg_list")

        self.mlp_P = tf.Variable(
            tf.random_normal([self.num_user, self.num_factor_mlp],
                             mean=4,
                             stddev=1 / (self.num_factor_mlp**0.5),
                             dtype=tf.float32))

        self.mlp_Q = tf.Variable(
            tf.random_normal([self.num_item, self.num_factor_mlp],
                             mean=4,
                             stddev=1 / (self.num_factor_mlp**0.5),
                             dtype=tf.float32))

        mlp_user_latent_factor = tf.nn.embedding_lookup(
            self.mlp_P, self.one_batch_user_list)
        mlp_item_latent_factor = tf.nn.embedding_lookup(
            self.mlp_Q, self.one_batch_pos_neg_list)

        mlp_layer_1 = tf.layers.dense(
            inputs=tf.concat([mlp_user_latent_factor, mlp_item_latent_factor],
                             axis=1),
            units=self.ranker_layers[0],
            kernel_initializer=tf.random_normal_initializer(
                mean=1, stddev=1 / (self.ranker_layers[0]**0.5)),
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name='layer_1')

        mlp_layer_2 = tf.layers.dense(
            inputs=mlp_layer_1,
            units=self.ranker_layers[1],
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(
                mean=1, stddev=1 / (self.ranker_layers[1]**0.5)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name='layer_2')

        mlp_layer_3 = tf.layers.dense(
            inputs=mlp_layer_2,
            units=self.ranker_layers[2],
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(
                mean=1, stddev=1 / (self.ranker_layers[2]**0.5)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name='layer_3')

        mlp_layer_4 = tf.layers.dense(
            inputs=mlp_layer_3,
            units=self.ranker_layers[3],
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(
                mean=1, stddev=1 / (self.ranker_layers[3]**0.5)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name='layer_4')

        _MLP = tf.layers.dense(
            inputs=mlp_layer_3,
            units=self.hidden_dimension,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(
                mean=1, stddev=1 / ((self.hidden_dimension)**0.5)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name='layer_5')

        self.pred_y = tf.layers.dense(
            inputs=mlp_layer_4,
            units=1,
            #activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=self.reg_rate),
            name="pred_layer")

        self.pred_pos_y = tf.slice(tf.reshape(self.pred_y, [-1]), [0],
                                   [self.batch_size])
        self.pred_neg_y = tf.slice(tf.reshape(self.pred_y, [-1]),
                                   [self.batch_size], [self.batch_size])

        self.hinge_ranking_loss = tf.reduce_sum(
            tf.maximum(
                tf.negative(
                    tf.subtract(self.pred_pos_y,
                                self.pred_neg_y + self.margin)), 0), 0)

        self.bpr_ranking_loss = tf.reduce_sum(
            tf.negative(
                tf.log(1e-10 + tf.sigmoid(
                    tf.subtract(self.pred_pos_y, self.pred_neg_y)))), 0)

        self.reg_loss = tf.losses.get_regularization_loss() + self.reg_rate * (
            tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))

        self.loss = self.bpr_ranking_loss + self.reg_loss

        global_step = tf.Variable(0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.starter_lr, beta1=0.9, beta2=0.999).minimize(self.loss, global_step=global_step) 

        print("network building finished.")

    def prepare_data(self, train_u_input, train_i_input, train_j_input,
                     train_labels, testItems, testRatings):
        """
        You must prepare the data before train and test the model.

        :param train_data:
        :param test_data:
        :return:
        """

        self.user = train_u_input
        self.item_i = train_i_input
        self.item_j = train_j_input
        self.label = train_labels

        self.testItems = testItems
        self.testRatings = testRatings

        self.num_training = len(train_u_input)
        self.total_batch = int(self.num_training / self.usual_batch_size)

        self.sampler = Sampler(user=self.user,
                               item_i=self.item_i,
                               item_j=self.item_j,
                               batch_size=self.usual_batch_size,
                               n_workers=1)

        print("data preparation finished.")

    def train(self):
        '''
        train one epoch
        '''
        self.epochs_Loss = []
        self.epochs_reg_Loss = []

        for batch_number in range(self.total_batch):
            one_batch_users, one_batch_item_i, one_batch_item_j = self.sampler.next_batch(
            )

            if len(one_batch_users) != self.batch_size:
                print(
                    "------------------------Warning!!!!!!!!!---------------------------------"
                )
                print("batch_number: ", batch_number)
                print("len(one_batch_users): ", len(one_batch_users))
                break

            one_batch_user_list = one_batch_users + one_batch_users
            one_batch_pos_neg_list = one_batch_item_i + one_batch_item_j

            pred_pos, pred_neg, _, loss, ranking_loss, reg_loss = self.sess.run(
                (self.pred_pos_y, self.pred_neg_y, self.optimizer, self.loss,
                 self.bpr_ranking_loss, self.reg_loss),
                feed_dict={
                    self.one_batch_user_list: one_batch_user_list,
                    self.one_batch_pos_neg_list: one_batch_pos_neg_list
                })

            self.epochs_Loss.append(np.sum(loss))
            self.epochs_reg_Loss.append(np.sum(reg_loss))

            if batch_number % self.display_step == 0:
                if self.verbose:
                    print("          batch: {}".format(batch_number))
                    print("          pred_pos_y: ", pred_pos)
                    print("          pred_neg_y: ", pred_neg)
                    print("          score difference: ", pred_pos - pred_neg)
                    print("          results: ",
                          [1 if dif > 0 else 0 for dif in pred_pos - pred_neg])

    def test(self, epoch):
        return evaluate_model(self, epoch)

    def execute(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)

        now = time()

        init_metrics = evaluate_model(self, -1)
        best_metrics = init_metrics
        log_file = 'results/deep/%d_%s_top%d.txt' % (now, self.dataset, self.topK)

        with open(log_file, 'a') as log:
            print('epoch %d: ' % -1, init_metrics)
            print('%d' % -1,
                    ' '.join('%.4f' % i for i in init_metrics),
                    file=log)

            for epoch in range(self.epochs):

                print("Begin training for epoch: %04d; " % epoch, end=' ')
                self.train()
                print("Finsh training for epoch: %04d; " % epoch)

                if epoch % self.T == 0:
                    print("Begin testings for epoch: %04d; " % epoch, end='')
                    metrics = self.test(epoch)
                    important_index = 1
                    if metrics[important_index] > best_metrics[important_index]:
                        best_metrics = metrics
                        print('epoch %d: ' % epoch, metrics, '[best]')
                        print('%d' % epoch,
                                ' '.join('%.4f' % i for i in metrics),
                                np.mean(self.epochs_Loss),
                                '[best]',
                                file=log)
                        self.save(path='models/NCPL_%s_model' % self.dataset,
                                    global_step=epoch)
                    else:
                        print('epoch %d: ' % epoch, metrics)
                        print('%d' % epoch,
                                ' '.join('%.4f' % i for i in metrics),
                                np.mean(self.epochs_Loss),
                                file=log)

        self.sampler.close()
        self.sess.close()

    def save(self, path, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, path, global_step)

    def predict(self, user_id, item_id):
        return self.sess.run(
            [self.pred_y],
            feed_dict={
                self.one_batch_user_list: user_id,
                self.one_batch_pos_neg_list: item_id
            })[0]
