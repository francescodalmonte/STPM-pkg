import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F



def L2dist_norm(array1, array2):
    """L2 distance between normalized features maps."""
    array1_norm = F.normalize(array1, p=2) # features vectors are L2 normalized at each "pixel" position
    array2_norm = F.normalize(array2, p=2)

    L2dist = 0.5 * ((array1_norm - array2_norm)**2).sum(axis=1)

    return L2dist



def total_loss(t_features, s_features, batch_avg=True):
    """Compute the total loss by comparing teacher's and student's
    output features (used L2dist_norm).

    If batch_avg == True this function returns a single value representing
    the average over all the images in the batch (used during training);
    otherwise, the output is a list of image-wise average loss values.
    """

    tot_loss = 0 if batch_avg else []

    N = len(t_features)
    for tf, sf in zip(t_features, s_features):
        b, c, h, w = tf.shape

        loss = L2dist_norm(tf, sf)

        if batch_avg:
            loss = loss.mean(dim=(0,1,2))
            tot_loss+=loss

        else:
            loss = loss.mean(dim=(1,2))
            tot_loss.append(loss.detach().cpu().numpy())

    if batch_avg:
        tot_loss/=N
    else:
        tot_loss = np.array(tot_loss)
        tot_loss = np.mean(tot_loss, axis=0)

    return tot_loss



def compute_anomaly_maps(t_features,
                         s_features,
                         out_size: int = 224):
    """Compute anomaly maps by interpolating the features-distance maps
    at different levels of the "pyramid"."""

    anomaly_maps = []

    for tf, sf in zip(t_features, s_features):
        loss = L2dist_norm(tf, sf)
        anomaly_map = F.interpolate(loss.unsqueeze(1),
                                    size=out_size,
                                    mode='bilinear',
                                    align_corners=False
                                    )

        anomaly_maps.append(anomaly_map.squeeze().detach().cpu().numpy())

    anomaly_maps = np.array(anomaly_maps).mean(axis=0)

    return anomaly_maps



def train_step(model_t,
               model_s,
               dataloader,
               device,
               optimizer,
               log_interval=-1):
    """Train the model for one epoch."""

    loss_super = []
    n_samples_super = []

    model_s.train() # only student model is trained

    for idx_batch, (x, y, mask) in enumerate(dataloader):
        time0 = time.time()
        n_samples = len(y)

        # forward pass
        optimizer.zero_grad()
        x = x.to(device)
        features_t = model_t(x)
        features_s = model_s(x)
        loss = total_loss(features_s, features_t)

        # backward pass
        time1 = time.time()
        loss.backward()
        optimizer.step()

        time2 = time.time()

        # log and store current values
        n_samples_super.append(n_samples)
        loss_super.append(loss.detach().cpu().numpy()*n_samples)

        if log_interval>0:
            if idx_batch%log_interval==0:
                time3 = time.time()
                print(f"TRAIN batch {idx_batch}/{len(dataloader)} - loss: {loss} - time: {time3-time0:.4f} s (fwd: {time1-time0:.4f} s, bwd: {time2-time1:.4f} s, other: {time3-time2:.4f} s)")

    return {"avg_loss": np.sum(loss_super)/np.sum(n_samples_super)}



def val_step(model_t,
             model_s,
             dataloader,
             device):
    """Single model validation step."""

    loss_super = []
    n_samples_super = []

    model_s.eval()

    with torch.no_grad():
        for idx_batch, (x, y, mask) in enumerate(dataloader):
            n_samples = len(y)

            # forward pass
            x = x.to(device)
            features_t = model_t(x)
            features_s = model_s(x)
            loss = total_loss(features_s, features_t)

            # log and store current values
            n_samples_super.append(n_samples)
            loss_super.append(loss.detach().cpu().numpy()*n_samples)

    return {"avg_loss": np.sum(loss_super)/np.sum(n_samples_super)}




def train_loop(model_t,
               model_s,
               train_loader,
               val_loader,
               device,
               num_epochs,
               optimizer,
               name_train,
               save_to,
               log_interval=-1,
               lr_scheduler=None,
               verbose=True):
    """Executes the training-evaluation loop."""
    print("Training loop started")
    
    if not os.path.isdir(os.path.join(save_to,'checkpoints')):
        os.mkdir(os.path.join(save_to, 'checkpoints'))

    losses_train = []
    losses_val = []

    best_val = np.inf

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # train step
        train_dict = train_step(model_t, model_s, train_loader,
                                device, optimizer, log_interval)

        # val step
        val_dict = val_step(model_t, model_s, val_loader, device)

        # store results
        losses_train.append(train_dict["avg_loss"])
        losses_val.append(val_dict["avg_loss"])

        lr = optimizer.param_groups[0]['lr']

        if lr_scheduler is not None:
            lr_scheduler.step()

        # save checkpoint if perfosmances improved on val set
        if val_dict["avg_loss"] < best_val:
            best_val = val_dict["avg_loss"]
            torch.save(model_s.state_dict(), os.path.join(save_to, f"checkpoints/{name_train}.ckpt"))
            msg = " (**ckpt)"
        else:
            msg = " "

        if verbose:
            elapsed = time.time()-start_time
            print(f"Epoch: {epoch} - TRAIN loss: {train_dict['avg_loss']:.6f} - VAL loss: {val_dict['avg_loss']:.6f} - LR: {lr:.6f}" , end=" - ")
            print(f"elapsed time: {elapsed:.4f} s {msg}")

    return {"losses_train": losses_train,
            "losses_val": losses_val
            }



def test_student_model(model_t,
                       model_s,
                       dataloader,
                       device):
    """Student model test."""

    inputs = []
    labels = []
    masks = []
    losses = []
    anomaly_maps = []

    model_s.eval()
    with torch.no_grad():
        for idx_batch, (x, y, mask) in enumerate(dataloader):

            # forward pass
            x = x.to(device)
            features_t = model_t(x)
            features_s = model_s(x)

            a_map = compute_anomaly_maps(features_s, features_t)
            a_map = gaussian_filter(a_map, sigma=3, axes=(1,2))

            inputs.extend(x.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())
            masks.extend(mask.squeeze().detach().cpu().numpy())
            anomaly_maps.extend(a_map)

    return {"inputs": np.array(inputs),
            "masks": np.array(masks),
            "labels": np.array(labels),
            "anomaly_maps": np.array(anomaly_maps),
            "avg_anomaly": np.mean(anomaly_maps, axis=(1,2)),
            "anomaly_peak": np.max(anomaly_maps, axis=(1,2))
            }