
def generate_config():
    return {
        # train.py related
        "db_root": "./all_image",
        "train_db": "./all_image/label_01_train.txt",
        "test_db": "./all_image/label_01_test.txt",

        "learning_rate": 1e-4,
        "optimizer": "rms",
        "momentum": 0.9,

        "epochs": 50,
        "batch_size": 3,
        "learning_rate_decay": [100, 150],
        "decay": 0.1,

        "loss_function": "npcc",
        "eval_metric": "npcc",

        "input_size": [256, 256, 1],
        "model_initialize_weight": './weight/epoch_17_-0.727022',
        "model_save_dir": './weight',

        # eval.py related
        "model_trained_weight": './weight/epoch_17_-0.727022',
        "eval_result_save_dir": 'eval_res',
        "text_save_filename": 'eval_result_01.txt',
    }