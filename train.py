import os
import torch
import tarfile
import argparse
from torch.utils.data import DataLoader
from sp4asc.datasets.dcase import DCaseDataset, DCaseDataset2
from sp4asc.models import get_net
from sp4asc.models.cnns import LogMelSpectrogram
from sp4asc.training import TrainingManager, TrainingManagerKD


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        for file in os.listdir(source_dir):
            if file == "data":
                pass
            elif file.split(".")[-1] == "egg-info":
                pass
            else:
                tar.add(os.path.join(source_dir, file))


if __name__ == "__main__":

    # --- Args
    parser = argparse.ArgumentParser(description="Training with mixup")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.py",
        help="Path to config file describing training parameters",
    )
    args = parser.parse_args()

    # ---
    print("Training script: ", os.path.realpath(__file__))

    # --- Config
    name_config = args.config.replace(".py", "").replace("/", ".")
    print(name_config)
    config = __import__(name_config, fromlist=["config"]).config
    print("Config parameters:")
    print(config)

    

    # --- Log dir
    path2log = config["out_dir"] + name_config
    os.makedirs(path2log, exist_ok=True)
    #make_tarfile(path2log + "/src.tgz", os.path.dirname(os.path.realpath(__file__)))
    print("hello")

    # ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    '''train_dataset = DCaseDataset(
        current_dir + "/data/TAU-urban-acoustic-scenes-2020-mobile-development/",
        split="train",
    )'''
    print("hello")
    train_dataset2 = DCaseDataset2(
        os.path.join(current_dir , "data/TAU-urban-acoustic-scenes-2020-mobile-development/").replace("\\", "/"),
        split="train",
    )

    '''test_dataset = DCaseDataset(
        current_dir + "/data/TAU-urban-acoustic-scenes-2020-mobile-development/",
        split="val",
    )'''

    test_dataset2 = DCaseDataset2(
        os.path.join(current_dir , "data/TAU-urban-acoustic-scenes-2020-mobile-development/").replace("\\", "/"),
        split="val",
    )

    '''loader_train = DataLoader(
        train_dataset,
        batch_size=config["batchsize"],
        shuffle=True,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=config["batchsize"],
        shuffle=False,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=False,
    )'''

    loader_train2 = DataLoader(
        train_dataset2,
        batch_size=config["batchsize"],
        shuffle=True,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    loader_test2 = DataLoader(
        test_dataset2,
        batch_size=config["batchsize"],
        shuffle=False,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    # --- Get network
    spectrogram = LogMelSpectrogram()
    net = get_net[config["net"]](
        config["dropout"],
        config["specAugment"],
    )
    net_c = get_net[config["net_c"]](
        config["dropout"],
        config["specAugment"],
    )
    print("\n\nNet at training time")
    print(net)
    print("Nb. of parameters at training time: ", net.get_nb_parameters() / 1e3, "k")

    # ---
    optim = torch.optim.AdamW(
        [
            {"params": net.parameters()},
        ],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        config["max_epoch"],
        eta_min=config["eta_min"],
    )


    # --- Training
    '''mng = TrainingManager(
        net,
        spectrogram,
        loader_train,
        loader_test,
        optim,
        scheduler,
        config,
        path2log,
    )
    #mng.train()'''

    mngkd = TrainingManagerKD(
        net,
        net_c,
        spectrogram,
        loader_train2,
        loader_test2,
        optim,
        scheduler,
        config,
        path2log,
    )
    mngkd.load_state()
    mngkd.train()
