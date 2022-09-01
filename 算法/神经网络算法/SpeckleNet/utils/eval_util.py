import numpy as np
from keras.losses import binary_crossentropy,mean_squared_error
import keras.backend as K
import keras_contrib.backend as KC

# evaluation
class npcc():
    def __init__(self, weight=None, reduction=True,size_average=True):
        super(npcc, self).__init__()
        self.reduce = reduction

    def __call__(self, pred, target):
        target = target.flatten()
        pred = pred.flatten()

        vpred = pred - pred.mean()
        vtarget = target - target.mean()

        cost = - (vpred * vtarget).sum() / \
               (np.sqrt((vpred ** 2).sum())
                * np.sqrt((vtarget ** 2).sum()) + 1e-8)

        # print(cost)
        if self.reduce is True:
            return cost.mean()
        return cost


eval_dict = {
    'npcc': npcc(),
    'bce': binary_crossentropy,
    'mse': mean_squared_error
}


def generate_eval_function(loss_name):
    return eval_dict[loss_name]


if __name__ == '__main__':
    x = np.random.rand(1, 28, 28)
    y = np.random.rand(1, 28, 28)
    a = npcc(x, y)
    print(a(x, y))
