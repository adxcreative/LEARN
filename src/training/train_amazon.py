import logging
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from tools.metric_tools import Ranker, AverageMeterSet


def is_master(args):
    return args.rank == 0


def construct_classification_loss(user_features, history_attention_mask,
                                  history_ids,
                                  item_features, gt_label,
                                  loss_func, logit_scale=20, ):
    # user_feat (bt, 40, emb_dim), item_feat (num_gallery, 1, emb_dim)
    # import pdb; pdb.set_trace()
    feature_dim = user_features.shape[-1]
    history_last_idx = history_attention_mask.sum(dim=1, keepdim=True) - 1  # (bt, 1)

    expand_first_random_index = history_last_idx.unsqueeze(dim=2).repeat(1, 1,
                                                                         feature_dim)  # # (bt, 1) -> (bt, 1, emb_dim)
    selected_user_features = torch.gather(user_features, dim=1, index=expand_first_random_index)
    selected_user_features = selected_user_features.squeeze()  # (bt, emb_dim)
    item_features = item_features.squeeze()  # (num_gallery, 1, emb_dim) -> (num_gallery, emb_dim)

    cosine = torch.matmul(selected_user_features, item_features.t())

    if history_ids is not None:
        mask = torch.zeros((cosine.size(0), cosine.size(1) + 1), device=cosine.device)
        # last dim is for padding
        mask = mask.scatter_(1, history_ids, - 10000.0)[:, :-1]
        cosine = cosine + mask

    # cross entropy loss
    logits = cosine * logit_scale
    loss = loss_func(logits, gt_label)
    loss = loss.mean()

    pred_idx = torch.argsort(logits, dim=1, descending=True)
    # calculate recall
    f2s_acc = torch.sum(pred_idx[:, 0] == gt_label) / gt_label.shape[0]
    return loss, f2s_acc


def construct_contrastive_loss(user_features, user_content_emb,
                               history_attention_mask,
                               item_features, item_content_emb,
                               target_itemid,
                               loss_func,
                               args,
                               logit_scale=20,
                               ):
    # import pdb; pdb.set_trace()
    batch_size, _, feature_dim = user_features.shape

    # =================== history sequence ===================
    user_last_index = history_attention_mask.sum(dim=1, keepdim=True) - 1  # (bs, 1)
    # (bs, 10) -> (bs, 10, 4096)
    expand_user_last_index = user_last_index.unsqueeze(dim=2).repeat(1, 1, feature_dim)
    selected_user_feat = torch.gather(user_features, dim=1, index=expand_user_last_index)
    # (bs, 1, 4096) -> (bs, 4096)
    selected_user_feat = selected_user_feat.squeeze()

    selected_user_content = torch.gather(user_content_emb, dim=1, index=expand_user_last_index)
    selected_user_content = selected_user_content.squeeze()

    # (bt, 1, emb_dim) -> (bt, emb_dim)
    item_features = item_features.squeeze()

    expand_contrast_id = target_itemid.unsqueeze(0).repeat(batch_size, 1)
    contrast_mask = expand_contrast_id == target_itemid.reshape(-1, 1)
    diag_matrix = torch.diag(torch.ones(batch_size)).cuda(args.local_device_rank, non_blocking=True)
    contrast_mask = (contrast_mask * 1. - diag_matrix) * -10000.0
    sim_matrix = torch.matmul(selected_user_feat, item_features.t())
    logit_per_pair = logit_scale * sim_matrix
    logit_per_pair = logit_per_pair + contrast_mask

    ground_truth = torch.arange(batch_size).long()

    # user loss mask to add ignore index -255

    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
    # CrossEntropy reduction='none'
    u2i_loss = loss_func(logit_per_pair, ground_truth)
    u2i_loss = u2i_loss.mean()

    # u2i_acc
    pred = logit_per_pair.argmax(-1)
    u2i_acc = (pred == ground_truth).sum() / len(logit_per_pair)
    return u2i_loss, u2i_acc


def get_ft_loss(model,
                history_item_info_list, target_item_info_list,
                history_attention_mask, target_attention_mask,
                history_itemid, target_itemid,
                loss_first, args, item_embedding,
                ):
    # import pdb; pdb.set_trace()

    user_features, item_features = model(
        history_item_info_list,
        history_attention_mask,
        target_item_info_list,
        target_attention_mask,
        item_embedding
    )

    if args.train_stage == "pretrain":
        u2i_loss, u2i_acc = construct_contrastive_loss(
            user_features, history_item_info_list,
            history_attention_mask,
            item_features, target_item_info_list,
            target_itemid,
            loss_first,
            args,
        )
        total_loss = u2i_loss
        loss_dict = {"u2i": u2i_loss}
        acc = None
        if args.report_training_batch_acc:
            acc = {"u2i": u2i_acc}

    elif args.train_stage == "finetune":
        u2i_loss, u2i_acc = construct_classification_loss(
            user_features,
            history_attention_mask,
            history_itemid,
            item_features,
            target_itemid,
            loss_first,
        )

        total_loss = u2i_loss
        loss_dict = {"u2i": u2i_loss}
        acc = None
        if args.report_training_batch_acc:
            acc = {"u2i": u2i_acc}

    return total_loss, acc, loss_dict


def get_item_pretrain_embedding(model, data, args):
    model.eval()
    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    data_iter = iter(dataloader)

    batch = next(data_iter)
    history_item_info_list = batch["history_item_emb_list"]
    history_attention_mask = batch["history_attention_mask"]
    batchsize = len(history_item_info_list)

    # target_item_info_list = batch["target_item_emb_list"]
    target_attention_mask = batch["target_attention_mask"]
    # for each sample in batch, target embeddings are all the targets
    # shape (bt, num_gallery, emb_dim) -> (num_gallery, 1, emb_dim)
    # target_item_info_list = target_item_info_list[0].unsqueeze(1)

    if args.train_stage == "pretrain":
        target_item_info_list = batch["target_item_emb_list"]
        target_item_info_list = target_item_info_list.unsqueeze(1)
    elif args.train_stage == "finetune":
        target_item_info_list = dataloader.dataset.idx2embedding.unsqueeze(1)
        # shape (bt, num_gallery, 1) -> (num_gallery, 1)
        target_attention_mask = target_attention_mask[0]

    target_itemid = batch["target_gt"]

    if not isinstance(history_item_info_list, list):
        history_item_info_list = history_item_info_list.cuda(args.local_device_rank, non_blocking=True)
        target_item_info_list = target_item_info_list.cuda(args.local_device_rank, non_blocking=True)

    history_attention_mask = history_attention_mask.cuda(args.local_device_rank, non_blocking=True)
    target_attention_mask = target_attention_mask.cuda(args.local_device_rank, non_blocking=True)
    target_itemid = target_itemid.cuda(args.local_device_rank, non_blocking=True)

    with torch.no_grad():
        user_features, item_features, logit_scale = model(
            # user_features, item_features = model(
            history_item_info_list,
            history_attention_mask,
            target_item_info_list,
            target_attention_mask
        )

    return item_features.squeeze()


def train_amazon(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps, wb, item_embedding=None):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()

    dataloader, sampler = data["train"].dataloader, data["train"].sampler

    loss_first = nn.CrossEntropyLoss(ignore_index=-255, reduction='none')

    loss_first = loss_first.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq

    data_iter = iter(dataloader)

    end = time.time()
    epoch_trained_steps = 0
    for i in range(global_trained_steps - num_batches_per_epoch * epoch, num_batches_per_epoch):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info(
                f"Stopping training due to step {step} has reached max_steps {args.max_steps // args.accum_freq}")
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        history_item_info_list = batch["history_item_emb_list"]

        if batch.get("history_item_id_list", None) is not None:
            history_itemid = batch["history_item_id_list"]
            history_itemid = history_itemid.cuda(args.local_device_rank, non_blocking=True)
        else:
            history_itemid = None

        history_attention_mask = batch["history_attention_mask"]
        batchsize = len(history_item_info_list)

        target_attention_mask = batch["target_attention_mask"]
        if args.train_stage == "pretrain":
            target_item_info_list = batch["target_item_emb_list"]
            target_item_info_list = target_item_info_list.unsqueeze(1)
        elif args.train_stage == "finetune":
            target_item_info_list = dataloader.dataset.idx2embedding.unsqueeze(1)
            # shape (bt, num_gallery, 1) -> (num_gallery, 1)
            target_attention_mask = target_attention_mask[0]

        target_itemid = batch["target_gt"]

        if not isinstance(history_item_info_list, list):
            history_item_info_list = history_item_info_list.cuda(args.local_device_rank, non_blocking=True)
            target_item_info_list = target_item_info_list.cuda(args.local_device_rank, non_blocking=True)

        history_attention_mask = history_attention_mask.cuda(args.local_device_rank, non_blocking=True)
        target_attention_mask = target_attention_mask.cuda(args.local_device_rank, non_blocking=True)
        target_itemid = target_itemid.cuda(args.local_device_rank, non_blocking=True)

        data_time = time.time() - end

        m = model.module

        if args.precision == "amp":
            with autocast():
                total_loss, acc, loss_dict = get_ft_loss(model,
                                                         history_item_info_list, target_item_info_list,
                                                         history_attention_mask, target_attention_mask,
                                                         history_itemid, target_itemid,
                                                         loss_first, args, item_embedding
                                                         )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0 and wb != None:
            wb.log(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "learning rate": optimizer.param_groups[0]["lr"],
                    "loss": total_loss.item(),
                    "uer2item acc": acc["u2i"].item() * 100,
                    "uer2item loss": loss_dict["u2i"].item(),
                    "Data Time": data_time,
                    "Batch Time": batch_time,
                }
            )
        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = int(batchsize) * args.accum_freq
            num_samples = (i + 1) * batchsize * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i + 1) / num_batches_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                f"User2Item Acc: {acc['u2i'].item() * 100:.2f} | " +
                f"User2Item Loss: {loss_dict['u2i'].item():.6f} | " +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

        if (args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0):
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_step{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": total_loss.item(),
                }, save_path)
            logging.info(
                f"Saved checkpoint {save_path} (epoch {epoch + 1} @ {step + 1} steps) (writing took {time.time() - t1} seconds)")

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
            logging.info(
                f"Saved checkpoint {save_path} (epoch {epoch + 1} @ {step + 1} steps) (writing took {time.time() - t1} seconds)")

    return epoch_trained_steps


def evaluate_amazon(model, data, epoch, args, wb, item_embedding):
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.eval()
    if is_master(args):
        logging.info("Evaluating...")

    user_dataloader = data["test_user"].dataloader
    item_dataloader = data["test_item"].dataloader
    metric_ks = [10, 50, 200]

    ranker = Ranker(metric_ks)
    average_meter_set = AverageMeterSet()

    # get all item features
    if item_embedding is None:
        with torch.no_grad():
            item_feat_list = []
            itemid_list = []
            for i, batch in enumerate(item_dataloader):
                itemid = batch["itemid"]
                item_emb = batch['item_emb']
                item_emb = item_emb.cuda(args.local_device_rank, non_blocking=True)

                item_emb = item_emb.unsqueeze(1)
                item_features = model.module.extract_item_embedding(item_emb)

                item_feat_list.append(item_features)
                itemid_list.append(itemid)

            item_feat_list = torch.cat(item_feat_list, dim=0)
            itemid_list = torch.cat(itemid_list, dim=0)
    else:
        item_feat_list = item_embedding
    with torch.no_grad():
        user_feat_list = []
        user_gt = []
        for i, batch in enumerate(user_dataloader):

            userid = batch["userid"]
            history_gt = batch['history_gt']
            history_item_info_list = batch["history_item_emb_list"]
            history_attention_mask = batch["history_attention_mask"]

            if batch.get("history_item_id_list", None) is not None:
                history_ids = batch["history_item_id_list"]
                history_ids = history_ids.cuda(args.local_device_rank, non_blocking=True)
            else:
                history_ids = None
            batchsize = len(history_item_info_list)

            history_item_info_list = history_item_info_list.cuda(args.local_device_rank, non_blocking=True)
            history_attention_mask = history_attention_mask.cuda(args.local_device_rank, non_blocking=True)

            user_features = model.module.extract_user_embedding(None, history_item_info_list, history_attention_mask)
            pred_scores = torch.matmul(user_features, item_feat_list.T)  # * logit_scale.exp()

            res = ranker(pred_scores, history_gt, history_ids)

            metrics = {}
            for i, k in enumerate(metric_ks):
                metrics["NDCG@%d" % k] = res[2 * i]
                metrics["Recall@%d" % k] = res[2 * i + 1]
            metrics["MRR"] = res[-2]
            metrics["AUC"] = res[-1]

            for k, v in metrics.items():
                average_meter_set.update(k, v)

        average_metrics = average_meter_set.averages()

        if wb != None:
            wb.log(
                {
                    "epoch": epoch,
                    **average_metrics
                }
            )

        logging.info(f'Test set: {average_metrics}')
        return average_metrics
