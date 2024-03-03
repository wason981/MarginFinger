import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
import utils
import torch.nn.functional as F
import model_load
import matplotlib as mpl
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 设置随机种子以确保可重复性
torch.manual_seed(42)


def radar_chart(datas: list, mode: str = "our"):
    # 设置全局字体大小
    mpl.rcParams["font.size"] = 12
    categories =['MEP','MEL','MEA','FP','FT','TL']
    colors = {
        "Trigger": "#ff7f0e",
        "IPGuard": "#62a0cb",
        "CAE": "#4a7daf",
        "SAC-w": "#6bbd6b",
        "SAC-m": "#a286bc",
        "MF": "#d62728",
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(20, 10))
    for i, (field, data) in enumerate(datas.items()):
        for j, (mode, result) in enumerate(data.items()):
            ax = plt.subplot(2, 4, i * 4 + j + 1, polar=True)
            handles = []
            best_area = 0
            best_id=0
            for num,(key, values) in enumerate(result.items()):
                values += values[:1]
                x = [values[m] * np.sin(angles[m]) for m in range(len(values))]
                y = [values[m] * np.cos(angles[m]) for m in range(len(values))]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                if area > best_area:
                    best_area = area
                    best_id=num
                (line,) = ax.plot(
                    angles,
                    values,
                    "o-",
                    linewidth=1,
                    color=colors[key],
                    label=f"{key}:{area:.2f}",
                )
                handles.append(line)
                ax.fill(angles, values, alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), labels=categories)
            labels = [handle.get_label() for handle in handles]
            hspace = -0.27 if i == 0 else -0.2
            legend = ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, hspace),
                handlelength=0.9,
                columnspacing=0.1,
                ncol=3,
                fontsize=11,
            )
            legend.get_texts()[best_id].set_fontweight("bold")###加粗底行
            # legend.get_texts()[meta + 1].set_fontweight("bold")
            ax.set_title(f"{field} {mode}", fontweight="bold")
    plt.subplots_adjust(hspace=0.38, wspace=0.05)
    plt.savefig("./radar_chart.pdf", dpi=300)
    plt.show()


def denoise_image(image):
    print(image.shape)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    denoised_l_channel = cv2.fastNlMeansDenoising(
        l_channel, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    denoised_a_channels = cv2.fastNlMeansDenoising(
        a_channel, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    denoised_b_channels = cv2.fastNlMeansDenoising(
        b_channel, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    denoised_lab_image = cv2.merge(
        [denoised_l_channel, denoised_a_channels, denoised_b_channels]
    )
    denoised_rgb_image = cv2.cvtColor(denoised_lab_image, cv2.COLOR_LAB2RGB)
    return denoised_rgb_image


def squeeze_color_bit(image, i):
    fractional, integer = np.modf(image * (2**i - 1))
    image = integer / (2**i - 1)
    return image


def meta_adv_difference(k=2):
    model = model_load.load_model(0, "teacher", 'tiny')
    model.eval()

    for j, ft in enumerate(["EC", "EW", "NC", "NW"]):
        meta_data_set = utils.load_result(
            f"./fingerprint/original/our_meta_{ft}_dataset.pkl"
        )
        meta_outputs = F.softmax(model(meta_data_set["data"]), dim=1).detach()
        meta_y_1 = torch.topk(meta_outputs, k=k, dim=1)[0][:, 0].numpy()
        meta_y_2 = torch.topk(meta_outputs, k=k, dim=1)[0][:, 1].numpy()
        # meta_y_1 = torch.argmax(meta_outputs, dim=1).numpy()
        # meta_y_2 = torch.argmin(meta_outputs, dim=1).numpy()

        fake_data_set = utils.load_result(
            f"./fingerprint/original/our_fake_{ft}_dataset.pkl"
        )
        fake_outputs = F.softmax(model(fake_data_set["data"]), dim=1).detach()
        fake_y_1 = torch.topk(fake_outputs, k=k, dim=1)[0][:, 0].numpy()
        fake_y_2 = torch.topk(fake_outputs, k=k, dim=1)[0][:, 1].numpy()
        # fake_y_1 = torch.argmax(fake_outputs, dim=1).numpy()
        # fake_y_2 = torch.argmin(fake_outputs, dim=1).numpy()

        data = {
            "meta_y_1": meta_y_1,
            "meta_y_2": meta_y_2,
            "fake_y_1": fake_y_1,
            "fake_y_2": fake_y_2,
        }
        utils.save_result(f"./result/cv_{ft}_data.pkl", data)


def meta_adv_difference_helper():
    mpl.rcParams["font.family"] = "SimHei"
    mpl.rcParams["font.size"] = 16
    font = {"family": "SimHei", "size": 16}
    plt.figure(figsize=(18, 9))
    colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#1e76b3"]
    for i, mode in enumerate(["cv"]):#, "bci"
        path_dir = (
            "./result"
            if mode == "bci"
            else "/data/xuth/deep_ipr/IPR_CV/result"
        )
        path = os.path.join(path_dir, f"{mode}_EC_data.pkl")
        data = utils.load_result(path)
        x = len(data["meta_y_1"])
        meta_y_1 = data["meta_y_1"]
        meta_y_2 = data["meta_y_2"]
        adv_y_1 = data["fake_y_1"]
        adv_y_2 = data["fake_y_2"]
        ax = plt.subplot(2, 4, i * 4 + 1)
        ax.plot(
            range(1, x + 1),
            meta_y_1,
            color=colors[0],
            label="$\mathbf{X}_{meta}^{cc}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            meta_y_2,
            color=colors[1],
            label="$\mathbf{X}_{meta}^{cc}~top2$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_1,
            color=colors[2],
            label="$\mathbf{X}_{pert}^{cc}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_2,
            color=colors[3],
            label="$\mathbf{X}_{pert}^{cc}~top2$",
        )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_1,
        #     meta_y_1,
        #     where=(adv_y_1 > meta_y_1),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_2,
        #     meta_y_2,
        #     where=(adv_y_2 < meta_y_2),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.set_xlabel("$n$ samples")
        # ax.set_ylabel("probility")
        ax.set_xlabel("$n$个样本", fontdict=font)
        ax.set_ylabel("概率", fontdict=font)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{mode.upper()}", fontdict=font)
        ax.legend(loc="upper left")
        #########################################
        path = os.path.join(path_dir, f"{mode}_EW_data.pkl")
        data = utils.load_result(path)
        x = len(data["meta_y_1"])
        meta_y_1 = data["meta_y_1"]
        meta_y_2 = data["meta_y_2"]
        adv_y_1 = data["fake_y_1"]
        adv_y_2 = data["fake_y_2"]
        ax = plt.subplot(2, 4, i * 4 + 2)
        ax.plot(
            range(1, x + 1),
            meta_y_1,
            color=colors[0],
            label="$\mathbf{X}_{meta}^{cw}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            meta_y_2,
            color=colors[1],
            label="$\mathbf{X}_{meta}^{cw}~top2$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_1,
            color=colors[2],
            label="$\mathbf{X}_{pert}^{cw}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_2,
            color=colors[3],
            label="$\mathbf{X}_{pert}^{cw}~top2$",
        )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_1,
        #     meta_y_1,
        #     where=(adv_y_1 > meta_y_1),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_2,
        #     meta_y_2,
        #     where=(adv_y_2 < meta_y_2),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.set_xlabel("$n$ samples")
        # ax.set_ylabel("probility")
        ax.set_xlabel("$n$个样本", fontdict=font)
        ax.set_ylabel("概率", fontdict=font)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{mode.upper()}", fontdict=font)
        ax.legend(loc="upper left")
        #########################################
        path = os.path.join(path_dir, f"{mode}_NC_data.pkl")
        data = utils.load_result(path)
        x = len(data["meta_y_1"])
        meta_y_1 = data["meta_y_1"]
        meta_y_2 = data["meta_y_2"]
        adv_y_1 = data["fake_y_1"]
        adv_y_2 = data["fake_y_2"]
        ax = plt.subplot(2, 4, i * 4 + 3)
        ax.plot(
            range(1, x + 1),
            meta_y_1,
            color=colors[0],
            label="$\mathbf{X}_{meta}^{uc}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            meta_y_2,
            color=colors[1],
            label="$\mathbf{X}_{meta}^{uc}~top2$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_1,
            color=colors[2],
            label="$\mathbf{X}_{pert}^{uc}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_2,
            color=colors[3],
            label="$\mathbf{X}_{pert}^{uc}~top2$",
        )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_1,
        #     meta_y_1,
        #     where=(adv_y_1 > meta_y_1),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_2,
        #     meta_y_2,
        #     where=(adv_y_2 < meta_y_2),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.set_xlabel("$n$ samples")
        # ax.set_ylabel("probility")
        ax.set_xlabel("$n$个样本", fontdict=font)
        ax.set_ylabel("概率", fontdict=font)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{mode.upper()}", fontdict=font)
        ax.legend(loc="upper left")
        #########################################
        path = os.path.join(path_dir, f"{mode}_NW_data.pkl")
        data = utils.load_result(path)
        x = len(data["meta_y_1"])
        meta_y_1 = data["meta_y_1"]
        meta_y_2 = data["meta_y_2"]
        adv_y_1 = data["fake_y_1"]
        adv_y_2 = data["fake_y_2"]
        ax = plt.subplot(2, 4, i * 4 + 4)
        ax.plot(
            range(1, x + 1),
            meta_y_1,
            color=colors[0],
            label="$\mathbf{X}_{meta}^{uw}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            meta_y_2,
            color=colors[1],
            label="$\mathbf{X}_{meta}^{uw}~top2$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_1,
            color=colors[2],
            label="$\mathbf{X}_{pert}^{uw}~top1$",
        )
        ax.plot(
            range(1, x + 1),
            adv_y_2,
            color=colors[3],
            label="$\mathbf{X}_{pert}^{uw}~top2$",
        )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_1,
        #     meta_y_1,
        #     where=(adv_y_1 > meta_y_1),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.fill_between(
        #     range(1, x + 1),
        #     adv_y_2,
        #     meta_y_2,
        #     where=(adv_y_2 < meta_y_2),
        #     interpolate=True,
        #     alpha=0.5,
        #     color="#c58f9c",
        # )
        # ax.set_xlabel("$n$ samples")
        # ax.set_ylabel("probility")
        ax.set_xlabel("$n$个样本", fontdict=font)
        ax.set_ylabel("概率", fontdict=font)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{mode.upper()}", fontdict=font)
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("./result/prob-dist-cn.pdf")
    # plt.show()


def demo():
    # 大矩形的坐标和尺寸
    big_rect = plt.Rectangle((1, 1), 6, 4, color="blue", alpha=0.5, zorder=2)

    # 小矩形的坐标和尺寸
    small_rect = plt.Rectangle((3, 2), 2, 1, color="red", alpha=0.8, zorder=1)

    # 创建绘图对象
    fig, ax = plt.subplots()

    # 添加大矩形和小矩形到坐标轴
    ax.add_patch(big_rect)
    ax.add_patch(small_rect)

    # 设置坐标轴范围
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)

    plt.show()


# def ablation(
#     model_feature_path: str, verbose: bool = False, features: list = [0, 1, 2, 3]
# ):
#     with open(model_feature_path, mode="r") as file:
#         reader = csv.reader(file)
#         features = [[float(row[i]) for i in features] + [str(row[4])] for row in reader]
#
#     source_feature = np.array([row[:-1] for row in features if row[-1] == "teacher"])
#     irr_feature = np.array([row[:-1] for row in features if row[-1] == "irrelevant"])
#     pro_feature = np.array([row[:-1] for row in features if row[-1] == "student_kd"])
#     lab_feature = np.array([row[:-1] for row in features if row[-1] == "student"])
#     cif_feature = np.array([row[:-1] for row in features if row[-1] == "CIFAR10C"])
#     fp_feature = np.array([row[:-1] for row in features if row[-1] == "fine-pruning"])
#     ft_feature = np.array(
#         [row[:-1] for row in features if row[-1] == "finetune_normal"]
#     )
#     adv_feature = np.array([row[:-1] for row in features if row[-1] == "adv_train"])
#
#     def helper(input):
#         input = np.array(input)
#         simi_score = np.linalg.norm(input - source_feature[0], ord=2)
#         return simi_score
#
#     irr_simi = list(map(helper, irr_feature))
#     pro_simi = list(map(helper, pro_feature))
#     lab_simi = list(map(helper, lab_feature))
#     cif_simi = list(map(helper, cif_feature))
#     fp_simi = list(map(helper, fp_feature))
#     ft_simi = list(map(helper, ft_feature))
#     adv_simi = list(map(helper, adv_feature))
#     pro_auc = utils.calculate_auc(
#         list_a=pro_simi, list_b=irr_simi, method="OURS", mode="pro", show=True
#     )
#     lab_auc = utils.calculate_auc(
#         list_a=lab_simi, list_b=irr_simi, method="OURS", mode="lab", show=True
#     )
#     cif_auc = utils.calculate_auc(
#         list_a=cif_simi, list_b=irr_simi, method="OURS", mode="cif", show=True
#     )
#     fp_auc = utils.calculate_auc(
#         list_a=fp_simi, list_b=irr_simi, method="OURS", mode="fp", show=True
#     )
#     ft_auc = utils.calculate_auc(
#         list_a=ft_simi, list_b=irr_simi, method="OURS", mode="ft", show=True
#     )
#     adv_auc = utils.calculate_auc(
#         list_a=adv_simi, list_b=irr_simi, method="OURS", mode="adv", show=True
#     )
#     if verbose:
#         print(
#             "pro:",
#             pro_auc,
#             "lab:",
#             lab_auc,
#             "cif:",
#             cif_auc,
#             "fp:",
#             fp_auc,
#             "ft:",
#             ft_auc,
#             "adv:",
#             adv_auc,
#         )
#     auc_records = [pro_auc, lab_auc, cif_auc, fp_auc, ft_auc, adv_auc]
#     return sum(auc_records) / len(auc_records)


if __name__ == "__main__":
    ####ABP
    datas = {
        "CIFAR10": {
            'Normal': {
                'Trigger': [0.66, 0.82, 0.73, 1.0, 1.0, 1.0],
                'IPGuard': [0.58, 0.55, 0.31, 0.93, 1.0, 0.26],
                'CAE': [0.95, 0.89, 0.9, 0.98, 1.0, 0.84],
                'SAC-w': [0.95, 1.0, 1.0, 1.0, 1.0, 1.0],
                'SAC-m': [0.95, 1.0, 0.91, 1.0, 1.0, 1.0],
                'MF': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            },
            'Query Attack': {
                'Trigger': [0.57, 0.73, 0.42, 1.0, 1.0, 1.0],
                'IPGuard': [0.5, 0.38, 0.37, 0.29, 0.1, 0.27],
                'CAE': [1.0, 0.96, 0.98, 0.98, 1.0, 0.96],
                'SAC-w': [0.9, 0.96, 0.92, 0.85, 0.62, 0.2],
                'SAC-m': [0.93, 0.91, 0.85, 0.84, 0.65, 0.3],
                'MF': [1.0, 0.96, 0.84, 0.93, 1.0, 1.0]
            },
            'Input Smooth': {
                'Trigger': [0.64, 0.81, 0.69, 1.0, 1.0, 1.0],
                'IPGuard': [0.42, 0.74, 0.42, 0.77, 0.65, 0.2],
                'CAE': [0.95, 0.93, 0.94, 0.98, 1.0, 0.95],
                'SAC-w': [0.95, 1.0, 1.0, 1.0, 1.0, 1.0],
                'SAC-m': [0.95, 1.0, 0.95, 1.0, 1.0, 1.0],
                'MF': [0.92, 0.89, 0.48, 0.82, 0.98, 0.85]
            },
            'Feature Squeeze': {
                'Trigger': [0.58, 0.74, 0.57, 1.0, 1.0, 1.0],
                'IPGuard': [0.52, 0.49, 0.3, 0.88, 1.0, 0.22],
                'CAE': [0.94, 0.9, 0.88, 0.98, 1.0, 0.85],
                'SAC-w': [0.95, 1.0, 1.0, 1.0, 1.0, 1.0],
                'SAC-m': [0.95, 1.0, 0.95, 1.0, 1.0, 1.0],
                'MF': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        },
        "TINY_IMAGENET": {
            "Normal": {        #  MEP   MEL   MEA    FP   FT    TL
                "Trigger":      [0.76, 0.80, 0.48, 1.00, 1.00, 1.00],
                "IPGuard":      [0.55, 0.58, 0.49, 0.66, 0.87, 0.00],
                # "EWE":          [0.97, 0.97, 0.91, 0.87, 1.00, 1.00],
                "CAE":          [0.95, 0.89, 0.90, 0.98, 1.00, 0.84],
                "SAC-w":        [0.24, 0.26, 0.00, 0.60, 1.00, 0.68],
                "SAC-m":        [0.37, 0.36, 0.06, 0.60, 1.00, 0.75],
                "MF":           [0.99, 0.89, 0.66, 0.80, 1.00, 0.98]
            },
            "Query Attack": {
                "Trigger":      [0.66, 0.68, 0.55, 1.00, 1.00, 0.87],
                "IPGuard":      [0.30, 0.56, 0.62, 0.13, 0.03, 0.00],
                # "EWE":          [0.97, 0.97, 0.91, 0.87, 1.00, 1.00],
                "CAE":          [1.00, 0.96, 0.98, 0.98, 1.00, 0.96],
                "SAC-w":        [0.67, 0.86, 0.57, 0.80, 1.00, 0.00],
                "SAC-m":        [0.59, 0.77, 0.39, 0.79, 1.00, 0.05],
                "MF":           [0.83, 0.69, 0.35, 0.67, 1.00, 0.90],                #     "Pert": [1.0, 0.93, 1.0, 1.0, 1.0, 1.0],
            },
            "Input Smooth": {#  ['FT','FP',   MEL','MEP','MEA','TL']
                "Trigger":      [0.72, 0.78, 0.52, 0.98, 1.00, 1.00],
                "IPGuard":      [0.69, 0.76, 0.73, 0.60, 1.00, 0.00],
                # "EWE":          [0.97, 0.97, 0.91, 0.87, 1.00, 1.00],
                "CAE":          [0.95, 0.93, 0.94, 0.98, 1.00, 0.95],
                "SAC-w":        [0.03, 0.05, 0.00, 0.60, 1.00, 0.70],
                "SAC-m":        [0.28, 0.28, 0.03, 0.60, 1.00, 0.83],
                "MF":           [0.76, 0.44, 0.25, 0.76, 0.90, 0.35],   # "Pert": [1.0, 0.93, 1.0, 1.0, 1.0, 0.99],
            },
            "Feature Squeeze": {
                "Trigger":      [0.70, 0.77, 0.32, 0.99, 1.00 ,1.00],
                "IPGuard":      [0.60, 0.54, 0.40, 0.35, 0.58, 0.00],
                # "EWE":          [0.97, 0.97, 0.91, 0.87, 1.00, 1.00],
                "CAE":          [0.94, 0.90, 0.88, 0.98, 1.00, 0.85],
                "SAC-w":        [0.15, 0.16, 0.00, 0.60, 1.00, 0.60],
                "SAC-m":        [0.35, 0.37, 0.02, 0.60, 1.00, 0.69],
                "MF":           [0.96, 0.98, 0.47, 0.69, 1.00, 1.00],               # "Pert": [1.0, 0.93, 1.0, 1.0, 1.0, 0.96],
            },
        },
    }
    radar_chart(datas=datas, mode="original")

#d_analysis_tiny
    # X =       [0.10,0.20,0.30,0.50,0.70,0.90]  # X轴坐标数据
    # extp =    [0.93,1.00,0.94,0.90,0.85,0.60] # Y轴坐标数据
    # extl =    [0.80,0.85,0.92,0.74,0.66,0.47] # Y轴坐标数据
    # extadv =  [0.58,0.63,0.62,0.38,0.16,0.00] # Y轴坐标数据
    # fp =      [0.79,0.80,0.80,0.80,0.80,0.75] # Y轴坐标数据
    # ft =      [1.00,1.00,1.00,1.00,1.00,1.00] # Y轴坐标数据
    # tl =      [0.71,0.75,0.85,0.96,1.00,1.00] # Y轴坐标数据
    # avg =     [0.80,0.84,0.86,0.80,0.75,0.64] # Y轴坐标数据
    # # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)
    # plt.plot(X, extp, label='MEP', marker='o',color="#3682be",linestyle='--')
    # plt.plot(X, extl, label='MEL', marker='o',color="#844bb3",linestyle='--')
    # plt.plot(X, extadv, label='MEA', marker='o',color="#eed777",linestyle='--')
    # plt.plot(X, fp, label='FP', marker='o',color="#334f65",linestyle='--')
    # plt.plot(X, ft, label='FT', marker='o',color="#ff7f0e",linestyle='--')
    # plt.plot(X, tl, label='TL', marker='o',color="#38cb7d",linestyle='--')
    # plt.plot(X, avg, label='AVG', marker='^',color="#d62728")
    # plt.xlabel(r"$d$")  # X轴标签
    # plt.ylabel("AUC")  # Y轴坐标标签
    # plt.title("Tiny-ImageNet $d$ Analysis")  # 图标题
    # plt.legend()
    # plt.savefig("show/d_analysis_tiny.png")
    # plt.show()


    # X =    [0.50, 1.00, 2.00, 3.00,4.00,5.00,]  # X轴坐标数据
    # extp=  [0.42, 0.70, 0.87, 0.90,0.94, 0.94]
    # extl=  [0.24, 0.51, 0.74, 0.90,0.92, 0.90]
    # extadv=[0.00, 0.05, 0.40, 0.54,0.62, 0.59]
    # fp =   [0.60, 0.60, 0.72, 0.78,0.80, 0.81]
    # ft =   [1.00, 1.00, 1.00, 1.00,1.00, 1.00]
    # tl =   [0.98, 0.71, 0.87, 0.85,0.85, 0.45]
    # avg =  [0.62, 0.75, 0.84, 0.83,0.86, 0.79]
    # plt.plot(X, extp, label='MEP', marker='o',color="#3682be",linestyle='--')
    # plt.plot(X, extl, label='MEL', marker='o',color="#844bb3",linestyle='--')
    # plt.plot(X, extadv, label='MEA', marker='o',color="#eed777",linestyle='--')
    # plt.plot(X, fp, label='FP', marker='o',color="#334f65",linestyle='--')
    # plt.plot(X, ft, label='FT', marker='o',color="#ff7f0e",linestyle='--')
    # plt.plot(X, tl, label='TL', marker='o',color="#38cb7d",linestyle='--')
    # plt.plot(X, avg, label='AVG', marker='^',color="#d62728")
    # plt.xlabel(r"$\gamma$")  # X轴标签
    # plt.ylabel("AUC")  # Y轴坐标标签
    # plt.title(r"Tiny-ImageNet $\gamma$ Analysis")  # 图标题
    # plt.legend()
    # plt.savefig("show/gamma_analysis_tiny.png")
    # plt.show()



    # X =      [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90]
    # extp =   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.98]
    # extl =   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.91]
    # extadv = [0.97, 1.00, 0.97, 0.94, 0.94, 0.94, 0.87, 0.73]
    # fp =     [0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.86, 0.86]
    # ft =     [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # tl =     [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99]
    # avg =    [0.97, 0.98, 0.97, 0.97, 0.97, 0.97, 0.96, 0.91]
    # plt.plot(X, extp, label='MEP', marker='o',color="#3682be",linestyle='--')
    # plt.plot(X, extl, label='MEL', marker='o',color="#844bb3",linestyle='--')
    # plt.plot(X, extadv, label='MEA', marker='o',color="#eed777",linestyle='--')
    # plt.plot(X, fp, label='FP', marker='o',color="#334f65",linestyle='--')
    # plt.plot(X, ft, label='FT', marker='o',color="#ff7f0e",linestyle='--')
    # plt.plot(X, tl, label='TL', marker='o',color="#38cb7d",linestyle='--')
    # plt.plot(X, avg, label='AVG', marker='^',color="#d62728")
    # plt.xlabel(r"$d$")  # X轴标签
    # plt.ylabel("AUC")  # Y轴坐标标签
    # plt.title("CIFAR-10 $d$ Analysis")  # 图标题
    # plt.legend()
    # plt.savefig("show/d_analysis_cifar.png")
    # plt.show()

    # X =     [1.00, 2.00, 3.00, 4.00, 5.00, 6.00]
    # extp =  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # extl =  [0.99, 1.00, 1.00, 1.00, 1.00, 0.96]
    # extadv =[0.91, 0.94, 0.96, 0.99, 0.96, 0.90]
    # fp =    [0.87, 0.87, 0.87, 0.87, 0.87, 0.85]
    # ft =    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # tl =    [1.00, 1.00, 1.00, 1.00, 1.00, 0.96]
    # avg =   [0.96, 0.97, 0.97, 0.98, 0.97, 0.95]
    # plt.plot(X, extp, label='MEP', marker='o',color="#3682be",linestyle='--')
    # plt.plot(X, extl, label='MEL', marker='o',color="#844bb3",linestyle='--')
    # plt.plot(X, extadv, label='MEA', marker='o',color="#eed777",linestyle='--')
    # plt.plot(X, fp, label='FP', marker='o',color="#334f65",linestyle='--')
    # plt.plot(X, ft, label='FT', marker='o',color="#ff7f0e",linestyle='--')
    # plt.plot(X, tl, label='TL', marker='o',color="#38cb7d",linestyle='--')
    # plt.plot(X, avg, label='AVG', marker='^',color="#d62728")
    # plt.xlabel(r"$\gamma$")  # X轴标签
    # plt.ylabel("AUC")  # Y轴坐标标签
    # plt.title(r"CIFAR10 $\gamma$ Analysis")  # 图标题
    # plt.legend()
    # plt.savefig("show/gamma_analysis_cifar.png")
    # plt.show()