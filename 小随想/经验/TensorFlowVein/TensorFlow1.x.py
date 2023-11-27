# encoding:utf-8
import numpy as np
import tensorflow as tf

# 构建神经网络并训练，使模型拟合 y=x^2+1
# 创建输入数据与标签数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
y_data = np.square(x_data) + 1 + np.random.normal(0, 0.05, x_data.shape)

# 定义输入数据属性
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 定义模型层函数
def model_layer(inputs, in_size, out_size, activation_function=None):
    """
    :param inputs: 数据输入
    :param in_size: 输入大小
    :param out_size: 输出大小
    :param activation_function: 激活函数（默认没有）
    :return:output：数据输出
    """
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    w_mul_x_add_b = tf.matmul(inputs, weights) + biases

    # 根据是否有激活函数
    if activation_function is None:
        output = w_mul_x_add_b
    else:
        output = activation_function(w_mul_x_add_b)
    return output


# 定义一个隐藏层
hidden_layer1 = model_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义一个输出层
output_layer1 = model_layer(hidden_layer1, 10, 1)

# 定义全局变量初始化 (在计算图中且被存储的变量，tf.local_variables_initializer()是在计算图中但未被存储的变量)
init_weight = tf.global_variables_initializer()

# 定义损失函数
loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(ys - output_layer1), reduction_indices=[1])
)

# 定义训练过程
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# tf.Session()通过启动一个tf后端会话来处理定义的操作
# 执行全局变量初始化
session = tf.Session()
session.run(init_weight)

# 执行训练过程
for i in range(1000):
    session.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))

# 结果关闭会话
session.close()
