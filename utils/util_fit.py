import torch
from tqdm import tqdm


def fit_one_epoch(
    epoch,
    end_epoch,
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    train_epoch_step,
    val_epoch_step,
    cuda,
    scheduler,
    save_dir,
):

    train_loss = 0
    train_num = 0
    val_loss = 0
    val_num = 0

    with tqdm(
        total=train_epoch_step,
        desc=f"Epoch {epoch + 1}/{end_epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for train_imgs, train_labels in train_dataloader:
            with torch.no_grad():
                if cuda:
                    train_imgs = train_imgs.cuda()
                    train_labels = train_labels.cuda()

            model.train()
            optimizer.zero_grad()
            train_confs, train_locs = model(train_imgs)
            loss_location, loss_classification = criterion(
                train_locs, train_confs, train_labels
            )
            loss = loss_location + loss_classification
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_imgs.size(0)
            train_num += train_imgs.size(0)
            pbar.set_postfix(
                {
                    "train_loss": train_loss / train_num,
                    "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                }
            )
            pbar.update(1)

    with tqdm(
        total=val_epoch_step,
        desc=f"Epoch {epoch + 1}/{end_epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for val_imgs, val_labels in val_dataloader:
            with torch.no_grad():
                val_imgs = val_imgs.cuda()
                val_labels = train_labels.cuda()

            model.eval()
            val_confs, val_locs = model(val_imgs)
            loss_location, loss_classification = criterion(
                val_locs, val_confs, val_labels
            )
            loss = loss_location + loss_classification
            val_num += val_imgs.size(0)
            val_loss += loss.item() * val_imgs.size(0)
            pbar.set_postfix(
                {
                    "val_loss": val_loss / val_num,
                    "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                }
            )
            pbar.update(1)

    if scheduler is not None:
        scheduler.step()
    if epoch < 40 and epoch % 2 == 0:
        torch.save(
            model.state_dict(),
            save_dir
            + f"ep{epoch+1}-train{train_loss/train_num}-val{val_loss/val_num}.pth",
        )
    if epoch >= 40:
        torch.save(
            model.state_dict(),
            save_dir
            + f"ep{epoch+1}-train{train_loss/train_num}-val{val_loss/val_num}.pth",
        )
    print(f"TRAIN_LOSS:{train_loss / train_num};VAL_LOSS:{val_loss / val_num}")
