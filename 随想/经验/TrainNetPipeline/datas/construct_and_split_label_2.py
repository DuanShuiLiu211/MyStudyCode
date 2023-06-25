import sys
import os
import random


def file_remove(file_list):
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass
    try:
        file_list.remove('._.DS_Store')
    except ValueError:
        pass


if __name__ == '__main__':
    """一对一的数据标签"""
    # 数据集名称
    data_folder = 'data'
    label_folder = 'label'

    # 标签文件名称
    all_sample_txt = 'label.txt'

    # 训练集划分
    train_ratio = 0.8
    k = 10

    # 打标签模式
    random_mode = 'all'
    if random_mode == 'k-fold':
        all_random = False
        k_fold_random = True
    elif random_mode == 'all':
        all_random = True
        k_fold_random = False
    else:
        print('Error! Please set random_mode k-fold or all')
        sys.exit()

    # 读取全部数据文件对象
    input_folderList = os.listdir(data_folder)
    file_remove(input_folderList)
    label_list = os.listdir(label_folder)
    file_remove(label_list)

    print(f'The number of labels is: {len(label_list)}.')

    # 构造匹配数据对
    sample_all_dict = {}
    for file in input_folderList:
        sample_list = []
        if os.path.isdir(file):
            data_list = os.listdir(os.path.join(data_folder, file))
            print(f'The number of {file} is: {len(data_list)}.')

            for datas_fn in data_list:
                if datas_fn in label_list:
                    sample_list.append('{} {}\n'.format(os.path.join(data_folder, file, datas_fn),
                                                        os.path.join(label_folder, datas_fn)))
                else:
                    print(f'Error! The {datas_fn} in folder: {file} cannot be found in folder: {label_folder}.')
                    sys.exit()
        else:
            if file in label_list:
                sample_list.append('{} {}\n'.format(os.path.join(data_folder, file),
                                                    os.path.join(label_folder, file)))
        sample_all_dict[f'{file}'] = sample_list

    # 将匹配数据对写入标签文档记录
    with open(all_sample_txt, 'w') as f:
        for _, sample_list in sample_all_dict.items():
            f.writelines(sample_list)

    # 全部随机
    if all_random:
        train_list = []
        test_list = []
        verify_list = []
        sample_len = len(list(sample_all_dict.values())[0])
        all_sample_len = len(list(range(sample_len * len(list(sample_all_dict.values())))))
        index_list = list(range(all_sample_len))
        random.shuffle(index_list)
        all_sample_list = []

        for i in range(len(list(sample_all_dict.values()))):
            all_sample_list.extend(list(sample_all_dict.values())[i])

        train_index = index_list[:int(all_sample_len * train_ratio)]
        test_index = index_list[int(all_sample_len * train_ratio):]

        train_list.extend([all_sample_list[idx] for idx in train_index])
        test_list.extend([all_sample_list[idx] for idx in test_index])
        verify_list.extend([all_sample_list[idx] for idx in test_index])

        with open('label_train.txt', 'w') as f:
            f.writelines(train_list)
        with open('label_test.txt', 'w') as f:
            f.writelines(test_list)
        with open('label_verify.txt', 'w') as f:
            f.writelines(verify_list)
        print(f'The label_train.txt, label_test.txt and label_verify.txt have been constructed')

    # k折随机
    if k_fold_random:
        indexs = locals()
        train_lists = locals()
        train_txts = locals()
        test_lists = locals()
        test_txts = locals()
        verify_lists = locals()
        verify_txts = locals()
        sample_len = len(list(sample_all_dict.values())[0])
        for folder_name, sample_list in sample_all_dict.items():
            index_list = list(range(sample_len))
            file = int(folder_name.split(',')[1])
            random.shuffle(index_list)

            for i in range(k):
                start = int(sample_len * i / k)
                end = int(sample_len * (i + 1) / k)
                indexs[f'index_{file}_{i + 1}'] = index_list[start: end]
                train_lists[f'trainfn_{k}_{i + 1}'] = []
                test_lists[f'testfn_{k}_{i + 1}'] = []
                verify_lists[f'verifyfn_{k}_{i + 1}'] = []

        for i in range(k):
            for folder_name, sample_list in sample_all_dict.items():
                file = int(folder_name.split(',')[1])
                for j in range(k):
                    if j != i:
                        train_lists[f'trainfn_{k}_{i + 1}'].extend([sample_list[idx] for idx in indexs[f'index_{file}_{j + 1}']])
                test_lists[f'testfn_{k}_{i + 1}'].extend([sample_list[idx] for idx in indexs[f'index_{file}_{i + 1}']])
                verify_lists[f'verifyfn_{k}_{i + 1}'].extend([sample_list[idx] for idx in indexs[f'index_{file}_{i + 1}']])

        for i in range(k):
            train_txts[f'label_train{i + 1}'] = 'label_train_{}.txt'.format(i + 1)
            test_txts[f'label_test{i + 1}'] = 'label_test_{}.txt'.format(i + 1)
            verify_txts[f'label_verify{i + 1}'] = 'label_verify_{}.txt'.format(i + 1)
            with open(train_txts[f'label_train{i + 1}'], 'w') as f:
                f.writelines(train_lists[f'trainfn_{k}_{i + 1}'])
            with open(test_txts[f'label_test{i + 1}'], 'w') as f:
                f.writelines(test_lists[f'testfn_{k}_{i + 1}'])
            with open(test_txts[f'label_verify{i + 1}'], 'w') as f:
                f.writelines(test_lists[f'verifyfn_{k}_{i + 1}'])
            print(f'The label_train{i + 1}.txt, label_test{i + 1}.txt and label_verify{i + 1}.txt have been constructed')
