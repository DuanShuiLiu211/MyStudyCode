import numpy as np
from keras.losses import binary_crossentropy, mean_squared_error
import keras.backend as K
import keras_contrib.backend as KC



class npccLoss():
    def __init__(self, reduction=True):
        super(npccLoss, self).__init__()
        self.reduce = reduction

    def __call__(self, pred, target):
        target = K.flatten(target)
        pred = K.flatten(pred)

        # import tensorflow as tf
        # with tf.Session():
        #     data1 = target.eval()
        #     data2 = pred.eval()

        vpred = pred - K.mean(pred)
        vtarget = target - K.mean(target)

        cost = - K.sum((vpred * vtarget)) / \
               (K.sum(K.sqrt((vpred ** 2)))
                * K.sum(K.sqrt((vtarget ** 2))) + 1e-8)

        if self.reduce is True:
            return K.mean(cost)
        return cost


class DSSIMObjective():
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5
    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.
    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        super(DSSIMObjective, self).__init__()
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, pred, target):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        target = K.reshape(target, [-1] + list(self.__int_shape(target)[1:]))
        pred = K.reshape(pred, [-1] + list(self.__int_shape(target)[1:]))

        patches_pred = KC.extract_image_patches(pred, kernel, kernel, 'valid',
                                                self.dim_ordering)
        patches_target = KC.extract_image_patches(target, kernel, kernel, 'valid',
                                                  self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_target = K.reshape(patches_target, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_target = K.mean(patches_target, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_target = K.var(patches_target, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_target_pred = K.mean(patches_target * patches_pred, axis=-1) - u_target * u_pred

        ssim = (2 * u_target * u_pred + self.c1) * (2 * covar_target_pred + self.c2)
        denom = ((K.square(u_target)
                  + K.square(u_pred)
                  + self.c1) * (var_pred + var_target + self.c2))
        ssim /= denom
        return K.mean((1.0 - ssim) / 2.0)


class joint_loss_ns():
    def __init__(self, *loss_fun_list):
        super(joint_loss_ns, self).__init__()
        self.loss_fun_list = [loss_fun() for loss_fun in loss_fun_list]

    def __call__(self, pred, target):
        out_loss = 0

        for loss_fun in self.loss_fun_list:
            temp_loss = loss_fun(pred, target)
            out_loss += temp_loss

        return out_loss


loss_dict = {
    'npcc' : npccLoss(),
    'ssim' : DSSIMObjective(),
    'ce' : binary_crossentropy,
    'mse' : mean_squared_error,
    'npcc_ssim' : joint_loss_ns(npccLoss, DSSIMObjective),

}


def generate_loss_function(loss_name):
    return loss_dict[loss_name]


if __name__ == '__main__':
    x = np.random.rand(1, 28, 28)
    y = np.random.rand(1, 28, 28)
    a = joint_loss_ns()
    print(a(x, y))

