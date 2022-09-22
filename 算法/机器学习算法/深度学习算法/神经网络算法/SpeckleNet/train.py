import os
import time
import keras.backend as K
from config import generate_config
from models.model import get_model_deep_speckle
from utils.data_util import generate_dataloader
from utils.eval_util import generate_eval_function
from utils.loss_util import generate_loss_function
from utils.opt_util import generate_optimizer
from utils.save_util import save_result_image


def train_keras(model, epoch, trainloader):
    totaLoss = 0
    sampleNum = 0
    for batch_idx, (dataList, target, fn) in enumerate(trainloader):
        loss = model.train_on_batch(dataList[0], target)
        totaLoss += loss
        sampleNum += len(dataList[0])

        if batch_idx % 1 == 0:
            outputString = '{} Train Epoch:{} [{}/{}] {}:{:.6f} LR:{:.10f}'.format(
                time.strftime('%Y-%m-%d', time.localtime(time.time())),
                epoch, sampleNum, len(trainloader.dataset), config['loss_function'],
                totaLoss / (batch_idx+1), K.get_value(model.optimizer.lr)
            )
            print(outputString)
            with open('train_log.txt', 'a') as f:
                f.write(outputString + '\n')

        save_result_image('train_image', epoch, fn, model.predict(dataList[0]))

    with open('train_loss.txt', 'a') as f:
        f.write('{}:{}\n'.format(epoch, totaLoss/(batch_idx+1)))


def test_keras(model, epoch, evaluator, testloader):
    totaLoss = 0
    for dataList, target, fn in testloader:
        output = model.predict(dataList[0])
        insloss = evaluator(output[:, :, :, 0], target[:, :, :, 0])
        totaLoss += insloss
        save_result_image('test_image', epoch, fn, output)

    avgLoss = totaLoss / len(testloader.dataset)
    outputString = 'Test avg {}:{:.6f} '.format(config['eval_metric'], avgLoss)
    print(outputString)

    with open('test_loss.txt', 'a') as f:
        f.write('{}:{}\n'.format(epoch, avgLoss))

    with open('train_log.txt', 'a') as f:
        f.write(outputString + '\n')

    save_model(config, model, epoch, avgLoss)

def adjust_learning_rate(optimizer, epoch, strategy):
    if epoch in strategy:
        K.set_value(optimizer.lr, K.get_value(optimizer.lr)*0.1)

def model_load_weight(config,model):
    try:
        model.load_weights(config['model_initialize_weight'], by_name=True)
        print('Successfully import weight from {} for model {}'.format(config['model_initialize_weight'], type(model).__name__))
    except Exception as r:
        print('Error：%s' %(r))
        print('Failed import weight from {} for model {}'.format(config['model_initialize_weight'], type(model).__name__))
    return model

def save_model(config, model, epoch, metric=None):
    save_path = config['model_save_dir']
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    if metric != None:
        model.save_weights(os.path.join(save_path, f'epoch_{epoch}_{metric:.6f}.h5'))
    else:
        model.save_weights(os.path.join(save_path, f'epoch_{epoch}_final.h5'))

if __name__ == '__main__':
    config = generate_config()
    epochs = config['epochs']
    model = get_model_deep_speckle(config)
    # model = get_model_deep_speckle_sam(config)
    # model = get_model_deep_speckle_adj(config)

    optimizer = generate_optimizer(config)

    criticizer = generate_loss_function(config['loss_function'])
    evaluator = generate_eval_function(config['eval_metric'])

    model.compile(optimizer, loss = criticizer)
    model = model_load_weight(config, model)

    train_dataloader, test_dataloader = generate_dataloader(config)

    for epoch_idx in range(epochs):
        train_keras(model, epoch_idx, train_dataloader)
        test_keras(model, epoch_idx, evaluator, test_dataloader)

        adjust_learning_rate(model.optimizer, epoch_idx+1, strategy=config['learning_rate_decay'])