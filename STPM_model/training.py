import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import torch
import torch.nn.functional as F



def L2dist_norm(array1, array2):
    """L2 distance between normalized features maps."""
    array1_norm = F.normalize(array1, p=2) # features vectors are L2 normalized at each "pixel" position
    array2_norm = F.normalize(array2, p=2)

    L2dist = 0.5 * ((array1_norm - array2_norm)**2).sum(axis=1)

    return L2dist

def cosine_dist(array1, array2):
    """Cosine distance between features maps."""
    array1_norm = F.normalize(array1, p=2) # features vectors are L2 normalized at each "pixel" position
    array2_norm = F.normalize(array2, p=2)

    cosine_dist = 1 - (array1_norm*array2_norm).sum(axis=1)

    return cosine_dist



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

        loss = cosine_dist(tf, sf)

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
    at different levels of the "pyramid".
    t/s_features shape : [n_pyramid_layers x b x c x h x w]"""

    anomaly_maps = []
    for tf, sf in zip(t_features, s_features):
        loss = L2dist_norm(tf, sf)
        anomaly_map = F.interpolate(loss.unsqueeze(1),
                                    size=out_size,
                                    mode='bilinear',
                                    align_corners=False
                                    )
        # anomaly_map shape : [b x 1 x out_size x out_size]

        anomaly_maps.append(anomaly_map.squeeze(1))
    anomaly_maps = torch.stack(anomaly_maps, dim=0)
    return anomaly_maps.mean(dim=0) # b x out_size x out_size



def train_step(model_t,
               model_s,
               out_features,
               dataloader,
               device,
               optimizer,
               log_interval=-1,
               lr_scheduler=None,
               limit_train_batches=np.inf,
               auxilary_dl=None):
    """Train the model for one epoch."""

    loss_super = []
    loss_aux_super = []
    loss_tot_super = []
    n_samples_super = []

    model_s.train() # only student model is trained

    for idx_batch, (x, y, mask) in enumerate(dataloader):
        if idx_batch>=limit_train_batches:
            break
        if auxilary_dl is not None:
            x_aux, _, _ = next(auxilary_dl)
            x_aux = x_aux.to(device)
            features_aux = model_s(x_aux, out_f=out_features)
            loss_aux = torch.stack([torch.pow(f, 2).mean() for f in features_aux]).mean()

        n_samples = len(y)

        # forward pass
        x = x.to(device)
        features_t = model_t(x, out_f=out_features)
        features_s = model_s(x, out_f=out_features)
        loss = total_loss(features_s, features_t)

        # backward pass
        optimizer.zero_grad()
        loss_tot = (loss + 0.1*loss_aux) if auxilary_dl is not None else loss
        loss_tot.backward()
        optimizer.step()

        # log and store current values
        n_samples_super.append(n_samples)
        loss_super.append(loss.detach()*n_samples)
        loss_aux_super.append(loss_aux.detach()*n_samples if auxilary_dl is not None else torch.tensor(torch.nan))
        loss_tot_super.append(loss_tot.detach()*n_samples)

        if log_interval>0:
            if idx_batch%log_interval==0:
                print(f"TRAIN batch {idx_batch}/{len(dataloader)}", end=" - ")
                print(f"loss: {loss_super[-1]/n_samples}", end=" - ")
                print(f"loss_aux: {loss_aux_super[-1]/n_samples}", end=" - ")
                print(f"loss_tot: {loss_tot_super[-1]/n_samples}", end="\r")
        if lr_scheduler is not None:
            lr_scheduler.step()
    loss_super = torch.stack(loss_super).cpu().numpy()
    loss_aux_super = torch.stack(loss_aux_super).cpu().numpy()
    loss_tot_super = torch.stack(loss_tot_super).cpu().numpy()

    return {"avg_loss": np.sum(loss_super)/np.sum(n_samples_super),
            "avg_loss_aux": np.sum(loss_aux_super)/np.sum(n_samples_super),
            "avg_loss_tot": np.sum(loss_tot_super)/np.sum(n_samples_super)
            }


def val_step(model_t,
             model_s,
             out_features,
             dataloader,
             device,
             log_interval=-1,
             auxilary_dl=None):
    """Single model validation step."""

    loss_super = []
    loss_aux_super = []
    loss_tot_super = []
    n_samples_super = []

    #model_s.eval()

    with torch.no_grad():
        for idx_batch, (x, y, mask) in enumerate(dataloader):
            if auxilary_dl is not None:
                x_aux, _, _ = next(auxilary_dl)
                x_aux = x_aux.to(device)
                features_aux = model_s(x_aux, out_f=out_features)
                loss_aux = torch.stack([torch.pow(f, 2).mean() for f in features_aux]).mean()
            
            n_samples = len(y)

            # forward pass
            x = x.to(device)
            features_t = model_t(x, out_f=out_features)
            features_s = model_s(x, out_f=out_features)
            loss = total_loss(features_s, features_t)

            # compute total loss
            loss_tot = (loss + 0.01*loss_aux) if auxilary_dl is not None else loss

            # log and store current values
            n_samples_super.append(n_samples)
            loss_super.append(loss*n_samples)
            loss_aux_super.append(loss_aux*n_samples if auxilary_dl is not None else torch.tensor(torch.nan))
            loss_tot_super.append(loss_tot*n_samples)

            if log_interval>0:
                if idx_batch%log_interval==0:
                    print(f"EVAL  batch {idx_batch}/{len(dataloader)}", end=" - ")
                    print(f"loss: {loss_super[-1]/n_samples}", end=" - ")
                    print(f"loss_aux: {loss_aux_super[-1]/n_samples}", end=" - ")
                    print(f"loss_tot: {loss_tot_super[-1]/n_samples}", end="\r")
                                                     
    loss_super = torch.stack(loss_super).cpu().numpy()
    loss_aux_super = torch.stack(loss_aux_super).cpu().numpy()
    loss_tot_super = torch.stack(loss_tot_super).cpu().numpy()

    return {"avg_loss": np.sum(loss_super)/np.sum(n_samples_super),
            "avg_loss_aux": np.sum(loss_aux_super)/np.sum(n_samples_super),
            "avg_loss_tot": np.sum(loss_tot_super)/np.sum(n_samples_super)
            }



def train_loop(model_t,
               model_s,
               out_features,
               train_loader,
               val_loader,
               device,
               num_epochs,
               optimizer,
               name_train,
               save_to,
               test_loader=None,
               log_interval=-1,
               lr_scheduler=None,
               limit_train_batches=np.inf,
               verbose=True,
               auxilary_dl=None):
    """Executes the training-evaluation loop."""
    print("Training loop started")
    
    if not os.path.isdir(os.path.join(save_to,'checkpoints')):
        os.mkdir(os.path.join(save_to, 'checkpoints'))

    losses_train = {"loss": [], "loss_aux": [], "loss_tot": []}
    losses_val = {"loss": [], "loss_aux": [], "loss_tot": []}
    LRs, AUROCs = [], []

    best_val = np.inf

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        # train step
        train_dict = train_step(model_t, model_s, out_features, train_loader,
                                device, optimizer, log_interval, 
                                lr_scheduler=lr_scheduler,
                                limit_train_batches=limit_train_batches,
                                auxilary_dl=auxilary_dl)

        # val step
        val_dict = val_step(model_t, model_s, out_features, val_loader,
                            device, log_interval,
                            auxilary_dl=auxilary_dl)
        
        # test model
        if test_loader is not None:
            test_dict = test_student_model(model_t, model_s, out_features,
                                           test_loader, device)
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(test_dict["labels"], test_dict["anomaly_peak"])
            roc_auc = auc(fpr, tpr)

        # store results
        losses_train["loss"].append(train_dict["avg_loss"])
        losses_train["loss_aux"].append(train_dict["avg_loss_aux"])
        losses_train["loss_tot"].append(train_dict["avg_loss_tot"])
        losses_val["loss_tot"].append(val_dict["avg_loss_tot"])
        losses_val["loss"].append(val_dict["avg_loss"])
        losses_val["loss_aux"].append(val_dict["avg_loss_aux"])

        lr = optimizer.param_groups[0]['lr']
        LRs.append(lr)
        if test_loader is not None:
            AUROCs.append(roc_auc)

        #if lr_scheduler is not None:
        #    lr_scheduler.step()

        # save checkpoint if performances improved on val set
        if val_dict["avg_loss_tot"] < best_val:
            best_val = val_dict["avg_loss_tot"]
            torch.save(model_s.state_dict(), os.path.join(save_to, f"checkpoints/{name_train}.ckpt"))
            msg = " (**ckpt)"
        else:
            msg = " "

        if verbose:
            elapsed = time.time()-start_time
            print(f"Epoch: {epoch}", end=" - ")
            print(f"TRAIN loss (reco/aux): {train_dict['avg_loss']:.6f}/{train_dict['avg_loss_aux']:.6f}", end=" - ")
            print(f"VAL loss (reco/aux): {val_dict['avg_loss']:.6f}/{val_dict['avg_loss_aux']:.6f}", end= " - ")
            if test_loader is not None:
                print(f"AUROC: {roc_auc:.5f}", end=" - ")
            print(f"LR: {lr:.6f}" , end=" - ")
            print(f"elapsed time: {elapsed:.4f} s {msg}")

    out_dict =  {
        "losses_train": losses_train,
        "losses_val": losses_val,
        "LRs": LRs
    }

    if test_loader is not None:
        out_dict["AUROCs"] = AUROCs
    
    return out_dict



def test_student_model(model_t,
                       model_s,
                       out_features,
                       dataloader,
                       device):
    """Student model test."""

    inputs = []
    labels = []
    masks = []
    anomaly_maps = []

    model_s.eval()
    with torch.no_grad():
        for idx_batch, (x, y, mask) in enumerate(dataloader):
            # forward pass
            x = x.to(device)
            features_t = model_t(x, out_f=out_features)
            features_s = model_s(x, out_f=out_features)

            a_map = compute_anomaly_maps(features_s, features_t)

            inputs.append(x.detach())
            labels.append(y.detach())
            masks.append(mask.squeeze().detach())
            anomaly_maps.append(a_map.detach())

        inputs = torch.cat(inputs, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        masks = torch.cat(masks, dim=0).cpu().numpy()
        anomaly_maps = torch.cat(anomaly_maps, dim=0).cpu().numpy()
        anomaly_maps = gaussian_filter(anomaly_maps, sigma=3, axes=(1,2))


    return {"inputs": np.array(inputs),
            "masks": np.array(masks),
            "labels": np.array(labels),
            "anomaly_maps": np.array(anomaly_maps),
            "avg_anomaly": np.mean(anomaly_maps, axis=(1,2)),
            "anomaly_peak": np.max(anomaly_maps, axis=(1,2))
            }