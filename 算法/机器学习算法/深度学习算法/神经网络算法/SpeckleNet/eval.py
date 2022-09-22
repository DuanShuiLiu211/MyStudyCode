from config import generate_config
from models.model import get_model_deep_speckle
from utils.data_util import generate_dataloader
from utils.eval_util import generate_eval_function
from utils.loss_util import generate_loss_function
from utils.opt_util import generate_optimizer
from utils.save_util import create_eval_dir
from utils.save_util import save_result_image_with_metric, save_results_in_file


def eval_keras(model, evaluator, testloader):
    totaLoss = 0
    fnList = []
    metricList = []

    save_dir = create_eval_dir(config['eval_result_save_dir'])

    for dataList, target, fn in testloader:
        output = model.predict(dataList[0])
        insloss = evaluator(output[:, :, :, 0],target[:, :, :, 0])
        totaLoss += insloss
        fnList.append(fn)
        metricList.append(insloss)
        save_result_image_with_metric(save_dir, fn, output, insloss)

    save_results_in_file(save_dir, config['text_save_filename'], fnList, metricList)

    avgLoss = totaLoss / len(testloader.dataset)
    outputString = 'Test avg {}:{:.6f} '.format('npcc', avgLoss)
    print(outputString)


def model_load_weight(config,model):
    try:
        model.load_weights(config['model_trained_weight'], by_name=True)
        print('Successfully import weight from {} for model {}'.format(config['model_trained_weight'], type(model).__name__))
    except Exception as r:
        print('Error：%s' %(r))
        print('Failed import weight from {} for model {}'.format(config['model_trained_weight'], type(model).__name__))
    return model


if __name__ == '__main__':
    config = generate_config()
    epochs = config['epochs']
    model = get_model_deep_speckle(config)

    optimizer = generate_optimizer(config)

    evaluator = generate_eval_function(config['eval_metric'])
    criticizer = generate_loss_function(config['loss_function'])

    model.compile(optimizer, loss = criticizer)
    model = model_load_weight(config, model)

    _, test_dataloader = generate_dataloader(config)

    eval_keras(model, evaluator, test_dataloader)