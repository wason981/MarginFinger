import os
import utils
import torch
import argparse
import model_load
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter


class DetectionAE(nn.Module):
    def __init__(self) -> None:
        super(DetectionAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Adding pooling layer
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class QueryAttack:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)

    def key_sample_detection(self, sample, threshold: float = 55, verbose: bool = True):
        model = DetectionAE()
        model.load_state_dict(torch.load(f"./model/autoencoder/model_best.pth"))
        model.eval()
        model.to(self.device)
        sample = torch.from_numpy(sample) if isinstance(sample, np.ndarray) else sample
        sample = torch.unsqueeze(sample, dim=0) if len(sample.shape) != 4 else sample
        sample = sample.to(self.device)
        total_loss = 0.0
        output = model(sample).detach()
        total_loss += torch.norm((output - sample), p=2).item()
        if verbose:
            return total_loss
        if total_loss >= threshold:
            return True
        else:
            return False

    def quary_attack(self, dataset):
        model = DetectionAE()
        model.load_state_dict(torch.load(f"./model/autoencoder/model_best.pth"))
        model.eval()
        model.to(self.device)
        inputs = []
        labels = []
        for sample in dataset:
            b_x = torch.unsqueeze(sample[0], dim=0).to(self.device)
            ae_output = model(b_x).detach()
            inputs.append(torch.squeeze(ae_output))
            labels.append(sample[1])
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return TensorDataset(inputs, labels)

    def visualize_cifar_sample(self, dataset, num, mode):
        """"""
        class_names = np.array(
            [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        )
        images = torch.stack([data[0].cpu() for data in dataset][:num])
        labels = torch.stack([data[1] for data in dataset][:num])

        # de_images = self.denormalize(images).numpy()
        #
        num_images = len(images)
        num_rows = int(np.ceil(num_images / 8))
        num_cols = min(num_images, 8)
        # 计算自适应的子图尺寸
        fig_width = 8
        fig_height = 1 * num_rows
        # 创建子图和设置标题
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                if index < num_images:
                    image = images[index]
                    label = labels[index]
                    axes[i][j].imshow(np.transpose(image, (1, 2, 0)))
                    axes[i][j].set_title(class_names[label], fontsize=10)
                axes[i][j].axis("off")
        # 去除多余的子图
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])
        plt.tight_layout(pad=1, h_pad=0.1, w_pad=2.5)
        os.makedirs(f"./query_attack/fake_data/AE/", exist_ok=True)
        plt.savefig(f"./query_attack/fake_data/AE/{mode}.png")
        plt.close(fig)

    def reconstruction_loss_by_datasets(self, method):
        if method == "train_ae":
            datas = utils.load_result(f"./query_attack/fingerprint(de)/ae_dataset.pkl")[
                "data"
            ]
        elif method == "empirical":
            datas = utils.load_result(
                f"./query_attack/fingerprint(de)/ae_noise_dataset.pkl"
            )["data"]
        elif method in [
            "our_fake_EC",
            "our_fake_EW",
            "our_fake_NC",
            "our_fake_NW",
            "our_meta_EC",
            "our_meta_EW",
            "our_meta_NC",
            "our_meta_NW",
        ]:
            datas = utils.denormalize(
                utils.load_result(f"./fingerprint/query_attack/{method}_dataset.pkl")[
                    "data"
                ]
            )
        elif method == "sac_m":
            datas = utils.denormalize(
                utils.load_result("./fingerprint/query_attack/sac_m_fingerprint.pkl")[
                    "data"
                ]
            )
        elif method == "sac_w":
            datas = utils.denormalize(
                utils.load_result("./fingerprint/query_attack/sac_w_fingerprint.pkl")[
                    "data"
                ]
            )
        elif method == "content_cifar":
            datas = utils.load_result("./data/query_attack/fingerprint(de).pkl")["data"]
        elif method == "cem":
            datas = utils.denormalize(
                utils.load_result("./fingerprint/query_attack/cem_fingerprint.pkl")[
                    "data"
                ]
            )
        elif method == "ipguard":
            datas = utils.denormalize(
                utils.load_result("./fingerprint/query_attack/ipguard_fingerprint.pkl")[
                    "data"
                ]
            )
        elif method == "trigger":
            datas = utils.denormalize(
                utils.load_result("./fingerprint/query_attack/trigger_wm.pkl")["data"]
            )
        total_loss = []
        for sample in datas:
            loss = self.key_sample_detection(sample=sample, threshold=50, verbose=True)
            total_loss.append(loss)
        print(sum(total_loss) / len(total_loss))


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.MSELoss,
    optimizer: Optimizer,
    verbose: bool = False,
):
    model.train()
    total_batches = len(data_loader)
    loss_record = []
    #
    for batch_idx, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        output = model(b_x)
        loss = loss_fn(output, b_x)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
        if verbose:
            print(f"loss: {loss:>7f}, [{batch_idx:>5d}/{total_batches:>5d}]")
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def val(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.MSELoss,
    verbose: bool = False,
):
    model.eval()
    total_batches = len(data_loader)
    loss_record = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            b_x = batch_data[0].to(device)
            output = model(b_x)
            loss = loss_fn(output, b_x)

            loss_record.append(loss.item())
            if verbose:
                print(f"loss: {loss:>7f}, [{batch_idx:>5d}/{total_batches:>5d}]")
    mean_val_loss = sum(loss_record) / total_batches
    return mean_val_loss


def fit(
    dataset: Dataset,
    model: nn.Module,
    args: argparse.ArgumentParser,
):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    save_dir = f"./fingerprint/query_attack/autoencoder/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    loss = float("inf")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch_id in tqdm(range(args.epochs)):
        train_loss = train(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            verbose=False,
        )

        writer.add_scalar("Loss/Train", train_loss, epoch_id)

        if train_loss < loss:
            loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))


def ours_dataset(ae, src_path, finger_type: str = "NC", fake: bool = True):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    path = (
        f"our_fake_{finger_type}_dataset.pkl"
        if fake
        else f"our_meta_{finger_type}_dataset.pkl"
    )
    utils.save_result(
        os.path.join("./fingerprint/query_attack/", path),
        {"data": data, "label": data_set["label"]},
    )


def query_attack_helper(ae, src_path, save_path):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


def ipguard_dataset(
    ae, src_path, save_path: str = "./fingerprint/query_attack/ipguard_fingerprint.pkl"
):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


def cem_dataset(
    ae, src_path, save_path: str = "./fingerprint/query_attack/cem_fingerprint.pkl"
):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


def sac_w_dataset(
    ae,
    src_path,
    save_path: str = "./fingerprint/query_attack/sac_w_fingerprint.pkl",
):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


def sac_m_dataset(
    ae,
    src_path,
    save_path: str = "./fingerprint/query_attack/sac_m_fingerprint.pkl",
):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


def trigger_dataset(
    ae,
    src_path,
    save_path: str = "./fingerprint/query_attack/trigger_wm.pkl",
):
    data_set = utils.load_result(src_path)
    data = torch.clamp(utils.denormalize(data_set["data"]), min=0.0, max=1.0)
    ae.to(device)
    ae.eval()
    data = utils.normalize(ae(data.to(device))).cpu()
    utils.save_result(save_path, {"data": data, "label": data_set["label"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    utils.seed_everything(2023)
    # make dataset
    # utils.ae_dataset()

    # train autoencoder
    # data_set = utils.load_result("./fingerprint/query_attack/ae_dataset.pkl")
    # dataset = TensorDataset(data_set["data"], data_set["label"])
    # model = DetectionAE()
    # fit(dataset=dataset, model=model, args=args)

    # query attack
    ae = model_load.load_model(0, "query_attack", 'tiny')

    # query_attack_helper(
    #     ae,
    #     src_path="./fingerprint/original/ipguard_fingerprint.pkl",
    #     save_path="./fingerprint/query_attack/ipguard_fingerprint.pkl",
    # )
    #
    # query_attack_helper(
    #     ae,
    #     src_path="./fingerprint/original/cem_fingerprint.pkl",
    #     save_path="./fingerprint/query_attack/cem_fingerprint.pkl",
    # )
    #
    # query_attack_helper(
    #     ae,
    #     src_path="./fingerprint/original/sac_w_fingerprint.pkl",
    #     save_path="./fingerprint/query_attack/sac_w_fingerprint.pkl",
    # )
    #
    # query_attack_helper(
    #     ae,
    #     src_path="./fingerprint/original/sac_m_fingerprint.pkl",
    #     save_path="./fingerprint/query_attack/sac_m_fingerprint.pkl",
    # )
    #
    # query_attack_helper(
    #     ae,
    #     src_path="./fingerprint/original/trigger_wm.pkl",
    #     save_path="./fingerprint/query_attack/trigger_wm.pkl",
    # )
    query_attack_helper(
        ae,
        src_path="./fingerprint/original/margin_fingerprint.pkl",
        save_path="./fingerprint/query_attack/margin_fingerprint.pkl",
    )
    # for ft in ["EC", "NC", "EW", "NW"]:
    #     ours_dataset(
    #         ae,
    #         src_path=f"./fingerprint/original/our_fake_{ft}_dataset.pkl",
    #         finger_type=ft,
    #         fake=True,
    #     )

    # for ft in ["EC", "NC", "EW", "NW"]:
    #     ours_dataset(
    #         ae,
    #         src_path=f"./fingerprint/original/our_meta_{ft}_dataset.pkl",
    #         finger_type=ft,
    #         fake=False,
    #     )

    qa = QueryAttack()
    # 1.9580243167757987
    # qa.reconstruction_loss_by_datasets(method="train_ae")
    # 5.332616149950027
    # qa.reconstruction_loss_by_datasets(method="empirical")
    # [0.6888815021514892, 0.7118647003173828, 0.710749265551567, 0.7016920638084412] avg:0.7033
    # for ft in ["EC", "EW", "NC", "NW"]:
    #     qa.reconstruction_loss_by_datasets(method=f"our_fake_{ft}")
    # [0.6981294333934784, 0.7118495529890061, 0.7107696169614792, 0.7017341738939286] avg:0.7056
    # for ft in ["EC", "EW", "NC", "NW"]:
    #     qa.reconstruction_loss_by_datasets(method=f"our_meta_{ft}")
    # 0.722934542298317
    # qa.reconstruction_loss_by_datasets(method="sac_m")
    # 0.7002885920642796
    # qa.reconstruction_loss_by_datasets(method="sac_w")
    # 0.8024453526735306
    # qa.reconstruction_loss_by_datasets(method="trigger")
    # 2.754135354757309
    # qa.reconstruction_loss_by_datasets(method="content_cifar")
    # 0.7190946478302739
    # qa.reconstruction_loss_by_datasets(method="cem")
    # 0.7259115898609161
    # qa.reconstruction_loss_by_datasets(method="ipguard")
