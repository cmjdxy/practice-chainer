# -*- coding: utf-8 -*-
"""
A new file.
"""
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training.extensions as E
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


plt.switch_backend('agg')


class ANN(chainer.Chain):
    def __init__(self, out_size, hiddens):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, hiddens[0])
            self.l2 = L.Linear(hiddens[0], hiddens[1])
            self.l3 = L.Linear(hiddens[1], out_size)
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


def main(args):
    # hyperparameters
    out_size = 10
    hiddens = [200, 50]
    learning_rate = 0.001
    batch_size = 1000
    n_epoch = 10
    log_dir = 'logs/ann'

    switch_log = 1

    # datasets
    if not args.manual:
        train_set, test_set = chainer.datasets.get_mnist() # this is OK
    else:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_set = chainer.datasets.TupleDataset(
            mnist.train.images, mnist.train.labels
        )
        test_set = chainer.datasets.TupleDataset(
            mnist.test.images, mnist.test.labels
        )

    train_iter = chainer.iterators.SerialIterator(
        train_set, batch_size, repeat=True, shuffle=True
    )
    test_iter = chainer.iterators.SerialIterator(
        test_set, batch_size, repeat=False, shuffle=False
    )

    # model
    model = L.Classifier(ANN(out_size, hiddens))

    # optimizer
    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)

    # trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(
        updater,
        stop_trigger=(n_epoch, 'epoch'),
        out=log_dir
    )

    # extension
    trainer.extend(E.Evaluator(test_iter, model))
    # ======= log ======= #
    if switch_log:
        trainer.extend(E.LogReport())
        trainer.extend(E.PrintReport(
            ['epoch',
             'main/loss',
             'validation/main/loss',
             'main/accuracy',
             'validation/main/accuracy',
             'elapsed_time']
        ))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: ANN')
    parser.add_argument('--manual', type=int, default=1,
                        help='1 for manual dataset, 0 for built-in one.')
    args = parser.parse_args()

    main(args)
