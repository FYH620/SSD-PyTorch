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
from utils.util_fit import fit_one_epoch


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def warmup_cosine_lr(epoch):
    if epoch < args.warmup_epoch:
        return (1 + epoch) / args.warmup_epoch
    else:
        return 0.95 ** (epoch - args.warmup_epoch + 1)


def cosine_unfreeze_lr(epoch):
    return (
        args.min_lr
        + 0.5
        * (args.unfreeze_lr - args.min_lr)
        * (1 + cos((epoch - args.unfreeze_epoch) / 8 * pi))
    ) / args.unfreeze_lr


parser = argparse.ArgumentParser(description="Configuration of training parameters.")
parser.add_argument(
    "--init_batchsize", default=32, type=int, help="Batch size for training."
)
parser.add_argument(
    "--unfreeze_batchsize",
    default=16,
    type=int,
    help="Batch size for unfreeze training.",
)
parser.add_argument(
    "--resume", default=False, type=str2bool, help="Whether to continue training."
)
parser.add_argument(
    "--resume_path", default=None, type=str, help="Resume model path for your training."
)
parser.add_argument("--warmup_epoch", default=5, type=int, help="Warmup epoch.")
parser.add_argument("--init_epoch", default=0, type=int, help="Start epoch.")
parser.add_argument("--unfreeze_epoch", default=40, type=int, help="Unfreeze epoch.")
parser.add_argument("--end_epoch", default=120, type=int, help="End epoch.")
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help=" The Number of workers used in dataloading.",
)
parser.add_argument(
    "--cuda", default=False, type=str2bool, help="Use CUDA to train model."
)
parser.add_argument(
    "--init_lr", default=1e-4, type=float, help="Initial learning rate."
)
parser.add_argument(
    "--unfreeze_lr", default=1e-5, type=float, help="Unfreeze learning rate."
)
parser.add_argument("--min_lr", default=1e-6, type=float, help="Min learning rate.")
parser.add_argument(
    "--save_folder",
    default="weights/",
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
        if step < 20:
            param.requires_grad = False

    if args.resume:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ssd_model.load_state_dict(torch.load(args.resume_path))
        print("Loading the resume weights {} done.".format(args.resume_path))
    else:
        ssd_model.loadPretrainedWeights()
        print("Loading the pretrained weights done.")

    if args.cuda:
        cudnn.benchmark = True
        ssd_model = ssd_model.cuda()

    print("Loading the dataset.")
    train_dataset = VOCDataset(mode="train", size=300, keep_difficult=False)
    val_dataset = VOCDataset(mode="val", size=300, keep_difficult=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.init_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.init_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate,
    )

    print("Initializing the optimizer,scheduler and the loss function.")
    optimizer = Adam(ssd_model.parameters(), lr=args.init_lr, weight_decay=5e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    criterion = MultiboxLoss(
        num_classes=21,
        overlap_threshold=0.5,
        neg_pos_ratio=3,
        use_cuda=args.cuda,
    )

    num_train = train_dataset.__len__()
    num_val = val_dataset.__len__()
    train_epoch_step = num_train // args.init_batchsize
    val_epoch_step = num_val // args.init_batchsize

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        fit_one_epoch(
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
        if step < 20:
            param.requires_grad = True
    optimizer = Adam(ssd_model.parameters(), lr=args.unfreeze_lr, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=cosine_unfreeze_lr)
    train_epoch_step = num_train // args.unfreeze_batchsize
    val_epoch_step = num_val // args.unfreeze_batchsize

    train_dataset = VOCDataset(mode="train", size=300, keep_difficult=False)
    val_dataset = VOCDataset(mode="val", size=300, keep_difficult=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.unfreeze_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.unfreeze_batchsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate,
    )

    for epoch in range(args.unfreeze_epoch, args.end_epoch):
        fit_one_epoch(
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
