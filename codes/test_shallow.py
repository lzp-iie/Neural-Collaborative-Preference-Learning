import argparse
import tensorflow as tf
import sys
import os.path
import os
tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shallow_cpl import Shallow_CPL
from my_utils import get_pairwise_train_dataset, get_test_data
from time import time


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model',
                        choices=[
                            'Shallow_CPL', 'Deep_CPL', 'NCPL'
                        ],
                        default='Shallow_CPL')

    parser.add_argument('--dataset',
                        default='ml100k')
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--reg_rate', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=1.9)
    parser.add_argument('--epochs', type=int, default=50)
    # batch_size: the number of different user_pos_item pair in one batch
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--t', type=int, default=1)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--num_factor', type=int, default=10)
    parser.add_argument('--num_factor_mlp', type=int, default=64)
    parser.add_argument('--hidden_dimension', type=int, default=10)
    parser.add_argument('--topK', type=int, default=10)
    parser.add_argument('--lr_decay_rate', type=float, default=0.85)
    parser.add_argument('--train_n', type=int, default=50)
    parser.add_argument('--test_n', type=int, default=70)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    margin = args.margin
    epoch = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    t = args.t
    display_step = args.display_step
    num_factor = args.num_factor
    num_factor_mlp = args.num_factor_mlp
    hidden_dimension = args.hidden_dimension
    topK = args.topK
    lr_decay_rate = args.lr_decay_rate
    n = args.train_n
    test_n = args.test_n

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    print(
        "-----------------------------learning_rate: {}------------------------------num_factor: {}---------------------------"
        .format(learning_rate, num_factor))
    print("!!!!!!!!!!!!!!!!Test_n: ", test_n, "!!!!!!!!!!!!!!!!!!!")                            
    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        model = None
        # Model selection
        if args.model == "Shallow_CPL":
            num_users, train_num_items, train_u_input, train_i_input, train_j_input = get_pairwise_train_dataset(
                path='data/%d_top%d/%s_train.dat' % (n, topK, dataset))

            print("Number of users: {} and train_items: {}".
                format(num_users, train_num_items))
            # num_users, num_items = 943, 1683
            # num_users, num_items = 9649, 17770

            
            train_labels = [1 for _ in range(len(train_u_input))]
            print("Length of train_u_input: %d " % len(train_u_input))

            train_labels = [1 for _ in range(len(train_u_input))]
            print("Length of train_u_input: %d " % len(train_u_input))
            if ('jester' in dataset or 'lastfm' in dataset):
                test_num_items, testItems, testRatings = get_test_data(path='data/%d_top%d/%s_test_ratings.lsvm' % (n, topK, dataset))
            else:
                test_num_items, testItems, testRatings = get_test_data(path='data/%d_top%d/%s_test_ratings_%d.lsvm' % (n, topK, dataset, test_n))
            print("Number of test_items: ", test_num_items)
            num_items = max(train_num_items, test_num_items)

            num_users += 1
            num_items += 1
            # the number of batch of one epoch: the number of total positive pairs / batch_size
            total_batch = int(
                len(train_u_input) /
                batch_size)

            model = Shallow_CPL(
                sess,
                dataset=dataset,
                num_user=num_users,
                num_item=num_items,
                learning_rate=learning_rate,
                reg_rate=reg_rate,
                margin=margin,
                epoch=epoch,
                batch_size=batch_size,
                verbose=verbose,
                t=t,
                display_step=display_step,
                num_factor=num_factor,
                num_factor_mlp=num_factor_mlp,
                hidden_dimension=hidden_dimension,
                topK=topK,
                lr_decay_rate=lr_decay_rate,
                test_n=test_n)

            model.prepare_data(train_u_input, train_i_input, train_j_input, train_labels, testItems, testRatings)

            model.build_network()

            model.execute()
