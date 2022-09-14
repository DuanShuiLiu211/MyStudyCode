# encoding:utf-8
import tensorflow as tf
from keras import Model
import matplotlib.pyplot as plt
import time


# 构建神经网络并训练，使模型对图片分类
# 第一阶段数据准备
# 导入数据集
fashion_mnist = tf.keras.datasets.fashion_mnist

label_class = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# 划分训练数据与测试数据
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据归一化 0~1 并结构化 [b w h c]
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[..., tf.newaxis].astype("float32")
test_images = test_images[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=60000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

for X, y in test_ds:
    print(f"Shape of X [N, H, W, C]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 数据可视化
visual_data = False
if visual_data:
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(label_class[train_labels[i]])
    plt.show()


# 第二阶段构建模型
# 定义模型类
class MyModel(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        output = x

        return output


# 实例化模型
model = MyModel()

# 定义优化器与损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 定义训练损失与准确性记录容器
train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 定义测试损失与准确性记录容器
test_loss_mean = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 计算训练集的梯度和损失
@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image, training=True)
        train_loss = loss_function(label, predictions)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_mean(train_loss)
    train_accuracy(label, predictions)

# 计算测试集的损失
@tf.function
def test_step(image, label):
    predictions = model(image, training=False)
    tt_ls = loss_function(label, predictions)

    test_loss_mean(tt_ls)
    test_accuracy(label, predictions)


if __name__ == "__main__":
    # 第三阶段运行策略
    num_epochs = 200
    with tf.device('/GPU:0'):
        for epoch in range(num_epochs):
            start_time = time.time_ns()
            train_loss_mean.reset_states()
            train_accuracy.reset_states()
            test_loss_mean.reset_states()
            test_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(images, labels)

            for test_images, test_labels in train_ds:
                test_step(test_images, test_labels)

            end_time = time.time_ns()
            print(
                f'Epoch {epoch + 1}, '
                f'Loss:{train_loss_mean.result() : .8f}, '
                f'Accuracy:{train_accuracy.result() * 100 : .2f}%, '
                f'Test Loss:{test_loss_mean.result() : .8f}, '
                f'Test Accuracy:{test_accuracy.result() * 100 : .2f}%, '
                f'Take time:{(end_time-start_time) / 1e9 : .4f}s'
            )
