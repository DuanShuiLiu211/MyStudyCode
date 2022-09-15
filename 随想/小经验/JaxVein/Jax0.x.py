from typing import Any, Callable, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
from tqdm.auto import tqdm
from functools import partial


# 定义模型
class ResNetBlock(nn.Module):
    """ResNet block."""
    ModuleDef = Any
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides,
                                 name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    ModuleDef = Any
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides,
                                 name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    ModuleDef = Any
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18_c10 = partial(ResNet,
                       stage_sizes=[2, 2, 2, 2],
                       block_cls=ResNetBlock,
                       num_classes=10)


# 定义数据流
def data_flow(*, dataset, batch_size=1, prng=None):
    total_data = len(dataset)
    if prng is not None:
        index_order = np.array(range(total_data))
        index_shuffle = jax.random.permutation(prng,
                                               index_order,
                                               independent=True)
    else:
        index_order = np.array(range(total_data))
        index_shuffle = index_order

    total_batch = total_data // batch_size
    for idx in range(total_batch):
        batch_index = index_shuffle[idx * batch_size:(idx + 1) * batch_size]
        mini_batch = [dataset[k] for k in batch_index]
        images = np.expand_dims(np.stack([x['image'] for x in mini_batch]),
                                -1).astype('float') / 255
        labels = np.stack([x['label'] for x in mini_batch])
        yield {'image': images, 'label': labels}


dataset_mnist = np.load("datasets/mnist.npy", allow_pickle=True).item()


# 定义损失函数
def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits,
                                       labels=labels_onehot).mean()


# 定义评估指标
def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


# 初始化模型进入训练状态
def create_train_state(*, prng, learning_rate, momentum):
    net = ResNet18_c10()
    params = net.init(prng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=net.apply,
                                         params=params,
                                         tx=tx)


# 定义训练方法
# 定义训练的每步操作
@jax.jit
def train_step(state, batch_data):
    """
    state: 不仅包含参数信息还包含优化器的信息等
    batch_data: 批数据 (N, H, W, C)
    """

    def loss_fn(params):
        logits, _ = ResNet18_c10().apply({'params': params},
                                         batch_data['image'],
                                         mutable=['batch_stats'])
        loss = cross_entropy_loss(logits=logits, labels=batch_data['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch_data['label'])
    return state, metrics


# 定义训练的执行逻辑
def train_model(state, epoch, batch_size, prng):
    batch_metrics = []
    train_dataset = dataset_mnist['train']
    total_batch = len(train_dataset) // batch_size

    with tqdm(data_flow(dataset=train_dataset,
                        batch_size=batch_size,
                        prng=prng),
              total=total_batch) as run_bar_set:
        for batch in run_bar_set:
            state, metrics = train_step(state, batch)
            batch_metrics.append(metrics)
            batch_metrics_jnp = jax.device_get(batch_metrics)
            epoch_metrics = {
                k: np.mean([metrics[k] for metrics in batch_metrics_jnp])
                for k in metrics.keys()
            }
            run_bar_set.set_description(
                f"train epoch: {epoch+1}, "
                f"loss: {epoch_metrics['loss']:.4f}, "
                f"accuracy: {(epoch_metrics['accuracy'] * 100):.2f}")

    return state


# 定义测试方法
# 定义测试的每步操作
@jax.jit
def test_step(params, batch_data):
    """
    params: 经过训练的参数
    batch_data: 批数据 (N, H, W, C)
    """
    logits, _ = ResNet18_c10().apply({'params': params},
                                     batch_data['image'],
                                     mutable=['batch_stats'])
    return compute_metrics(logits=logits, labels=batch_data['label'])


# 定义测试执行逻辑
def test_model(params, epoch, batch_size):
    batch_metrics = []
    test_dataset = dataset_mnist['test']
    total_batch = len(test_dataset) // batch_size

    with tqdm(data_flow(dataset=test_dataset, batch_size=batch_size),
              total=total_batch) as run_bar_set:
        for batch in run_bar_set:
            metrics = test_step(params, batch)
            batch_metrics.append(metrics)
            batch_metrics_jnp = jax.device_get(batch_metrics)
            epoch_metrics = {
                k: np.mean([metrics[k] for metrics in batch_metrics_jnp])
                for k in metrics.keys()
            }
            run_bar_set.set_description(
                f"train epoch: {epoch+1}, "
                f"loss: {epoch_metrics['loss']:.4f}, "
                f"accuracy: {(epoch_metrics['accuracy'] * 100):.2f}")

    return epoch_metrics


# 进行训练与测试
seed = 51
prng = jax.random.PRNGKey(seed)  # 通过种子获取随机数生成器密钥
prng, init_prng = jax.random.split(
    prng, 2)  # 拆分原随机数生成器密钥得到2个新的密钥，使用相同密钥随机函数将输出相同结果，用其实现可复现的权重初始化
num_epochs = 10
batch_size = 32
learning_rate = 0.1
momentum = 0.9
state = create_train_state(prng=init_prng,
                           learning_rate=learning_rate,
                           momentum=momentum)

for epoch in range(num_epochs):
    # 定义用于打乱数据顺序的伪随机数生成器
    prng, data_prng = jax.random.split(prng)
    for train_batch_data in data_flow(dataset=dataset_mnist['train'],
                                      batch_size=batch_size,
                                      prng=data_prng):
        print(train_batch_data['image'].shape, train_batch_data['image'].dtype)
        print(train_batch_data['label'].shape, train_batch_data['label'].dtype)
        break
    for test_batch_data in data_flow(dataset=dataset_mnist['test'],
                                     batch_size=batch_size):
        print(test_batch_data['image'].shape, test_batch_data['image'].dtype)
        print(test_batch_data['label'].shape, test_batch_data['label'].dtype)
        break
    # 训练模型
    state = train_model(state, epoch, batch_size, data_prng)
    # 测试模型
    params = state.params
    metrics = test_model(params, epoch, batch_size)

print("运行完成")
