import argparse
import warnings
import torch
import os
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.backends import cudnn
from torch.optim import Adam
from math import cos, pi
from nets.ssd import SSD
from utils.utils_dataload import VOCDataset
from utils.utils_multibox_loss import MultiboxLoss
from utils.utils_configuration import train_process_configuration as train_config
from utils.utils_configuration import voc_ssd300_configuration as ssd_config
from utils.utils_fit import fitOneEpoch


def strToBool(v):
    return v.lower() in ("yes", "true", "1")


def detectionCollateFunction(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def warmupCosineLearningRate(epoch):
    if epoch < args.warmup_epoch:
        return (1 + epoch) / args.warmup_epoch
    else:
        return 0.95 ** (epoch - args.warmup_epoch + 1)


def cosineUnfreezeLearningRate(epoch):
    return (
        args.min_lr
        + 0.5
        * (args.unfreeze_lr - args.min_lr)
        * (1 + cos((epoch - args.unfreeze_epoch) / 8 * pi))
    ) / args.unfreeze_lr


parser = argparse.ArgumentParser(description="Configuration of training parameters.")
parser.add_argument(
    "--init_batchsize",
    default=train_config["init_batchsize"],
    type=int,
    help="Batch size for training.",
)
parser.add_argument(
    "--unfreeze_batchsize",
    default=train_config["unfreeze_batchsize"],
    type=int,
    help="Batch size for unfreeze training.",
)
parser.add_argument(
    "--resume",
    default=train_config["resume_train"],
    type=strToBool,
    help="Whether to continue training.",
)
parser.add_argument(
    "--resume_path",
    default=train_config["resume_weights_path"],
    type=str,
    help="Resume model path for your training.",
)
parser.add_argument(
    "--warmup_epoch",
    default=train_config["warmup_epoch"],
    type=int,
    help="Warmup epoch.",
)
parser.add_argument(
    "--init_epoch",
    default=train_config["init_epoch"],
    type=int,
    help="Start epoch.",
)
parser.add_argument(
    "--unfreeze_epoch",
    default=train_config["unfreeze_epoch"],
    type=int,
    help="Unfreeze epoch.",
)
parser.add_argument(
    "--end_epoch",
    default=train_config["end_epoch"],
    type=int,
    help="End epoch.",
)
parser.add_argument(
    "--num_workers",
    default=train_config["num_workers"],
    type=int,
    help=" The Number of workers used in dataloading.",
)
parser.add_argument(
    "--cuda",
    default=train_config["use_cuda"],
    type=strToBool,
    help="Use CUDA to train model.",
)
parser.add_argument(
    "--init_lr",
    default=train_config["init_lr"],
    type=float,
    help="Initial learning rate.",
)
parser.add_argument(
    "--unfreeze_lr",
    default=train_config["unfreeze_lr"],
    type=float,
    help="Unfreeze learning rate.",
)
parser.add_argument(
    "--min_lr",
    default=train_config["min_lr"],
    type=float,
    help="Min learning rate.",
)
parser.add_argument(
    "--save_folder",
    default=train_config["save_weights_folder"],
    help="Directory for saving checkpoint models.",
)
args = parser.parse_args()
warnings.filterwarnings("ignore")


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    print("Generating the SSD detection model.")
    ssd_model = SSD(mode="train")
    ssd_model.initWeights()
    print("Freeze some layers to train.")
    for step, param in enumerate(ssd_model.parameters()):
        if step < train_config["num_freeze_layers"]:
            param.requires_grad = False

    if args.resume:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ssd_model.load_state_dict(torch.load(args.resume_path, map_location=device))
        print("Loading the resume weights {} done.".format(args.resume_path))
    else:
        ssd_model.loadPretrainedWeights()
        print("Loading the pretrained weights done.")

    if args.cuda:
        cudnn.benchmark = True
        ssd_model = ssd_model.cuda()

    print("Loading the dataset.")
    train_dataset = VOCDataset(
        mode="train",
        size=ssd_config["image_size"],
        keep_difficult=False,
    )
    val_dataset = VOCDataset(
        mode="val",
        size=ssd_config["image_size"],
        keep_difficult=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.init_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detectionCollateFunction,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.init_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detectionCollateFunction,
    )

    print("Initializing the optimizer,scheduler and the loss function.")
    optimizer = Adam(ssd_model.parameters(), lr=args.init_lr, weight_decay=5e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=warmupCosineLearningRate)
    criterion = MultiboxLoss(
        num_classes=ssd_config["num_classes"],
        overlap_threshold=ssd_config["positive_negative_iou_threshold"],
        neg_pos_ratio=ssd_config["negative_positive_ration"],
        use_cuda=args.cuda,
    )

    num_train = train_dataset.__len__()
    num_val = val_dataset.__len__()
    train_epoch_step = num_train // args.init_batchsize
    val_epoch_step = num_val // args.init_batchsize

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        fitOneEpoch(
            epoch=epoch,
            end_epoch=args.unfreeze_epoch,
            model=ssd_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            train_epoch_step=train_epoch_step,
            val_epoch_step=val_epoch_step,
            cuda=args.cuda,
            scheduler=scheduler,
            save_dir=args.save_folder,
        )

    print("The model has been unfreezed.")
    for step, param in enumerate(ssd_model.parameters()):
        if step < train_config["num_freeze_layers"]:
            param.requires_grad = True

    optimizer = Adam(ssd_model.parameters(), lr=args.unfreeze_lr, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=cosineUnfreezeLearningRate)
    train_epoch_step = num_train // args.unfreeze_batchsize
    val_epoch_step = num_val // args.unfreeze_batchsize

    train_dataset = VOCDataset(
        mode="train",
        size=ssd_config["image_size"],
        keep_difficult=False,
    )
    val_dataset = VOCDataset(
        mode="val",
        size=ssd_config["image_size"],
        keep_difficult=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.unfreeze_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detectionCollateFunction,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.unfreeze_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detectionCollateFunction,
    )

    for epoch in range(args.unfreeze_epoch, args.end_epoch):
        fitOneEpoch(
            epoch=epoch,
            end_epoch=args.end_epoch,
            model=ssd_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            train_epoch_step=train_epoch_step,
            val_epoch_step=val_epoch_step,
            cuda=args.cuda,
            scheduler=scheduler,
            save_dir=args.save_folder,
        )


if __name__ == "__main__":
    train()
