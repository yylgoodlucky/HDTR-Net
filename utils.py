import os
import torch

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_epoch{:02d}_step{:05d}.pth".format(epoch, step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "step": step,
        "epoch": epoch,
    }, checkpoint_path)
    
    print("Saved checkpoint:", checkpoint_path)