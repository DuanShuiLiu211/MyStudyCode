import os
import random

import cv2
import medpy.metric as medmetric
import numpy as np
import segmentation_models_pytorch as smp
import sklearn.metrics as skmetric
import torch
import xlrd
from torch.utils.data import DataLoader


def tes_process(model, test_loader, device):
    model.eval()
    img = []
    prob = []
    truth = []
    image_names = []

    metrics = smp.utils.metrics.Fscore()
    with torch.no_grad():
        for data in test_loader:
            inputs, label, image, image_name, origin_size = data
            inputs = inputs.to(device).float()
            outputs = model(inputs)

            outputs = outputs.data.cpu().numpy().squeeze()
            outputs = outputs.argmax(axis=0)
            image = image.data.cpu().numpy().squeeze()
            image = np.transpose(image, (1, 2, 0))
            label = label.numpy().squeeze()

            # resize回原尺寸
            image = cv2.resize(
                image,
                (origin_size[1].item(), origin_size[0].item()),
                interpolation=cv2.INTER_CUBIC,
            )
            outputs = cv2.resize(
                outputs,
                (origin_size[1].item(), origin_size[0].item()),
                interpolation=cv2.INTER_NEAREST,
            )
            label = cv2.resize(
                label,
                (origin_size[1].item(), origin_size[0].item()),
                interpolation=cv2.INTER_NEAREST,
            )

            img.append(image)
            prob.append(outputs)
            truth.append(label)
            image_names.append(image_name)

    return img, prob, truth, image_names


def result_show(i, img, pred, truth, image_name):
    cv_pred = np.zeros_like(img)
    cv_pred[pred == 1, 2] = 255
    cv_pred[pred == 2, 1] = 255

    cv_truth = np.zeros_like(img)
    cv_truth[truth == 1, 2] = 255
    cv_truth[truth == 2, 1] = 255

    img1 = cv2.addWeighted(img, 0.8, cv_pred, 0.2, 0)
    img2 = cv2.addWeighted(img, 0.8, cv_truth, 0.2, 0)

    cv2.imencode(".png", cv_pred)[1].tofile(
        os.path.join(save_dir, f"{image_name[0]}_pred.png")
    )
    cv2.imencode(".png", cv_truth)[1].tofile(
        os.path.join(save_dir, f"{image_name[0]}_label.png")
    )
    cv2.imencode(".png", img)[1].tofile(os.path.join(save_dir, f"{image_name[0]}.png"))


def cal_dice(i, pred, truth, spacing):
    pred_in = np.zeros_like(pred)
    truth_in = np.zeros_like(truth)
    pred_in[pred == 2] = 1
    truth_in[truth == 2] = 1

    dice_in = medmetric.dc(pred_in, truth_in)
    iou_in = medmetric.jc(pred_in, truth_in)
    recall_in = medmetric.recall(pred_in, truth_in)
    precision_in = medmetric.precision(pred_in, truth_in)

    pred_both = np.zeros_like(pred)
    truth_both = np.zeros_like(truth)
    pred_both[pred == 1] = 1
    truth_both[truth == 1] = 1
    pred_both[pred == 2] = 1
    truth_both[truth == 2] = 1

    dice_both = medmetric.dc(pred_both, truth_both)
    iou_both = medmetric.jc(pred_both, truth_both)

    recall_both = medmetric.recall(pred_both, truth_both)
    precision_both = medmetric.precision(pred_both, truth_both)

    if 0 == np.count_nonzero(pred_in):
        asd_in = assd_in = hd_in = hd_in95 = 5
    else:
        asd_in = medmetric.binary.asd(pred_in, truth_in, voxelspacing=spacing)
        assd_in = medmetric.binary.assd(pred_in, truth_in, voxelspacing=spacing)
        hd_in = medmetric.binary.hd(pred_in, truth_in, voxelspacing=spacing)
        hd_in95 = medmetric.binary.hd95(pred_in, truth_in, voxelspacing=spacing)

    if 0 == np.count_nonzero(pred_both):
        asd_both = assd_both = hd_both = hd_both95 = 5
    else:
        asd_both = medmetric.binary.asd(pred_both, truth_both, voxelspacing=spacing)
        assd_both = medmetric.binary.assd(pred_both, truth_both, voxelspacing=spacing)
        hd_both = medmetric.binary.hd(pred_both, truth_both, voxelspacing=spacing)
        hd_both95 = medmetric.binary.hd95(pred_both, truth_both, voxelspacing=spacing)

    # return dice_in, dice_both, iou_in, iou_both, recall_in, recall_both, precision_in, precision_both, asd_in,
    # asd_both, assd_in, assd_both, hd_in, hd_both, hd_in95, hd_both95
    return (
        100 * dice_in,
        100 * dice_both,
        100 * iou_in,
        100 * iou_both,
        100 * recall_in,
        100 * recall_both,
        100 * precision_in,
        100 * precision_both,
        asd_in,
        asd_both,
        assd_in,
        assd_both,
        hd_in,
        hd_both,
        hd_in95,
        hd_both95,
    )


if __name__ == "__main__":
    # parameters
    batch_size = 1
    num_workers = 0
    device_num = "4"

    # path
    model = r"efficientnet-b0_best.pt"
    root_dir = r"../data/原始390划出的测试集116_原始"
    save_dir = rf"./results/{model[:-3]}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_list = sorted(os.listdir(root_dir))
    print(f"test list len:{len(test_list)}")

    # dataset_loader
    print(f"Creating dataset...")
    test_set = Dataset(root_dir, test_list)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Now using {device}")

    model = torch.load(f"results/{model}", map_location=device)
    print(f"model loaded...")
    model.to(device)

    img, prob, truth, image_names = tes_process(model, test_loader, device)
    img_np = np.array(img)
    prob_np = np.array(prob)
    truth_np = np.array(truth)
    image_names_np = np.array(image_names)

    sdi = []
    sdb = []  # dice
    sii = []
    sib = []  # iou
    sri = []
    srb = []  # recall
    spi = []
    spb = []  # precision
    sasdi = []
    sasdb = []  # average surface distance
    sassdi = []
    sassdb = []  # average symmetric surface distance
    shdi = []
    shdb = []  # hausdorff distance
    shdi95 = []
    shdb95 = []  # hausdorff distance95

    spn = []
    stn = []

    excel = xlrd.open_workbook("spacing_all.xlsx")
    sheet = excel.sheet_by_index(0)
    rows = sheet.row_values(0)

    names = np.asarray(sheet.col_values(0))
    pixels = np.asarray(sheet.col_values(1))

    for i in range(len(prob)):
        spacing = int(float(pixels[np.argwhere(names == image_names[i])][0][0]))
        # di, do, db, ii, io, ib, pn, tn = cal_dice(i, prob[i], truth[i])

        (
            dice_in,
            dice_both,
            iou_in,
            iou_both,
            recall_in,
            recall_both,
            precision_in,
            precision_both,
            asd_in,
            asd_both,
            assd_in,
            assd_both,
            hd_in,
            hd_both,
            hd_in95,
            hd_both95,
        ) = cal_dice(i, prob[i], truth[i], 5 / spacing)

        result_show(i, img[i], prob[i], truth[i], image_names[i])

        sdi.append(dice_in)
        sdb.append(dice_both)
        sii.append(iou_in)
        sib.append(iou_both)
        sri.append(recall_in)
        srb.append(recall_both)
        spi.append(precision_in)
        spb.append(precision_both)

        sasdi.append(asd_in)

        sasdb.append(asd_both)

        sassdi.append(assd_in)

        sassdb.append(assd_both)

        shdi.append(hd_in)

        shdb.append(hd_both)

        shdi95.append(hd_in95)

        shdb95.append(hd_both95)

        # spn.append(pn)
        # stn.append(tn)

    # sdi = np.array(sdi)

    print(
        f"内：dice:{np.mean(sdi):.2f}±{np.std(sdi):.2f},iou:{np.mean(sii):.2f}±{np.std(sii):.2f},precision:{np.mean(spi):.2f}±{np.std(spi):.2f},recall:{np.mean(sri):.2f}±{np.std(sri):.2f},"
        f"asd:{np.mean(sasdi):.2f}±{np.std(sasdi):.2f},assd:{np.mean(sassdi):.2f}±{np.std(sassdi):.2f},hd:{np.mean(shdi):.2f}±{np.std(shdi):.2f},hd95:{np.mean(shdi95):.2f}±{np.std(shdi95):.2f}"
    )
    print(
        f"外：dice:{np.mean(sdb):.2f}±{np.std(sdb):.2f},iou:{np.mean(sib):.2f}±{np.std(sib):.2f},precision:{np.mean(spb):.2f}±{np.std(spb):.2f},recall:{np.mean(srb):.2f}±{np.std(srb):.2f},"
        f"asd:{np.mean(sasdb):.2f}±{np.std(sasdb):.2f},assd:{np.mean(sassdb):.2f}±{np.std(sassdb):.2f},hd:{np.mean(shdb):.2f}±{np.std(shdb):.2f},hd95:{np.mean(shdb95):.2f}±{np.std(shdb95):.2f}"
    )

    # from scipy.stats import ttest_rel
    #
    # res = ttest_rel(spn, stn)
    # spn = np.array(spn)
    # stn = np.array(stn)
