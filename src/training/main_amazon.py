from math import ceil
import os
import logging
import time
from time import strftime, localtime

import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from clip.LEARN import PreferenceAlignmentModule

from training.dataset_amazon import get_amazon18_pretrain, get_amazon18_finetune

from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr
import wandb

from training.train_amazon import train_amazon, evaluate_amazon


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return args.rank == 0


def get_dataloader(args, epoch_id=0):
    if args.train_stage == 'pretrain':
        data = get_amazon18_pretrain(
            args,
            epoch_id=epoch_id,
            max_one_year_length=args.max_one_year_length,
            max_one_month_length=args.max_one_month_length,
            sample_one_year_length=args.sample_one_year_length,
            sample_one_month_length=args.sample_one_month_length,
        )
    elif args.train_stage == 'finetune':
        data = get_amazon18_finetune(
            # data = get_ml_finetune(
            args,
            epoch_id=epoch_id,
            max_one_year_length=args.max_one_year_length,
            max_one_month_length=args.max_one_month_length,
            sample_one_year_length=args.sample_one_year_length,
            sample_one_month_length=args.sample_one_month_length,
        )
    else:
        assert False, f'train_stage {args.train_stage} not supported'

    return data


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )

    return trainable_params, all_param


def get_model(args):
    # import pdb; pdb.set_trace()
    model = PreferenceAlignmentModule(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )

    get_nb_trainable_parameters(model)

    return model


def main():
    args = parse_args()

    # Set distributed group
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_device_rank = max(local_rank, 0)
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)

    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    args.log_path = os.path.join(args.log_dir, args.name, "train_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.log_dir, args.name, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)

    # assert args.precision in ["amp", "fp16", "fp32"]

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)

    setup_worker_logging(args.rank, log_queue, args.log_level)

    model = get_model(args)
    if is_master(args):
        logging.info("Model: {}".format(model))

    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)

    if args.precision == 'bf16':
        model.to(dtype=torch.bfloat16).cuda(args.local_device_rank)
    else:
        model.cuda(args.local_device_rank)

    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_device_rank],
                                                      find_unused_parameters=False)
    if args.grad_checkpointing:
        model._set_static_graph()

    if args.precision == "fp16":
        convert_weights(model)

    # Initialize dataset and dataloader
    data = get_dataloader(args, epoch_id=0)

    # exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    # include = lambda n: not exclude(n)

    # named_parameters = list(model.named_parameters())
    # gain_or_bias_params = [
    #     p for n, p in named_parameters if exclude(n) and p.requires_grad
    # ]
    # embedding_params = [p for n, p in named_parameters if 'id_embedding' in n and p.requires_grad]
    # rest_params = [p for n, p in named_parameters if include(n) and 'id_embedding' not in n and p.requires_grad]
    get_nb_trainable_parameters(model)

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            params=model.parameters(),
            # [
            #     {"params": gain_or_bias_params, "weight_decay": 0.0, 'lr': args.lr},
            #     {"params": embedding_params, "weight_decay": args.wd, 'lr': args.lr},
            #     {"params": rest_params, "weight_decay": args.wd, 'lr': args.lr},
            # ],
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        num_batches = data["train"].dataloader.num_batches
        if args.max_steps is not None:
            args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
        else:
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
        total_steps = args.max_steps
        warmup_steps = int(num_batches // args.accum_freq) * args.warmup
        scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # Log and save hyper-params.
    if is_master(args) and not args.debug and args.wandb_project:
        wb = wandb.init(
            project=args.wandb_project,
            name=args.name,
            config=args,
        )
    else:
        wb = None
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.log_dir, args.name, "params_{}.txt".format(time_suffix))
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    if args.local_device_rank == 0:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    logging.info(f"Use GPU: {args.local_device_rank} for training")

    # Optionally resume from a checkpoint
    start_epoch = 0
    steps = 0

    if args.resume:
        if is_master(args):
            logging.info("=> begin to load checkpoint {}".format(args.resume))
        # if args.resume.split("epoch")[-1].find("_") + 1:
        #     epoch = int(args.resume.split("epoch")[-1].split("_")[1]) - 1
        #     step = int(
        #         args.resume.split("epoch")[-1].split("_")[2].split("step")[1].split(".")[0]
        #     )
        # else:
        #     epoch = int(args.resume.split("epoch")[-1].split(".")[0]) - 1
        #     step = 0000
        # if is_master(args):
        #     logging.info("epoch is %d, step is %d" % (epoch + 1, step))

        checkpoint = torch.load(args.resume, map_location="cpu")
        sd = checkpoint["state_dict"]
        lacked_key, unexpected_key = model.load_state_dict(sd, strict=False)
        # model_ema.module.load_state_dict(sd, strict=False)
        if is_master(args):
            logging.info("lacked_key is ")
            logging.info(lacked_key)
            logging.info("unexpected_key is ")
            logging.info(unexpected_key)
    else:
        logging.info("=> DO NOT loading any checkpoint")

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.should_save = ((args.log_dir is not None and args.log_dir != "" and args.log_dir.lower() != "none")
                        and is_master(args))

    item_embedding = None

    # for evaluation
    average_metrics = evaluate_amazon(model, data, 0, args, wb, item_embedding)

    for epoch in range(start_epoch, args.max_epochs):


        torch.distributed.barrier()

        num_steps_this_epoch = train_amazon(model, data, epoch, optimizer, scaler, scheduler, args, steps, wb,
                                            item_embedding)
        steps += num_steps_this_epoch

        if is_master(args):
            logging.info(f"Start epoch {epoch + 1}")
            average_metrics = evaluate_amazon(model, data, epoch, args, wb, item_embedding)

        # if exists next epoch, reload the dataset and dataloader for the next epoch
        if epoch + 1 < args.max_epochs:
            data = get_dataloader(args, epoch + 1)

        # Saving checkpoints.
        if args.should_save and num_steps_this_epoch > 0:
            if (epoch + 1) == args.max_epochs or (
                    args.save_epoch_frequency > 0
                    and ((epoch + 1) % args.save_epoch_frequency) == 0
            ):
                t1 = time.time()
                save_path = os.path.join(args.checkpoint_path, f"epoch{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": steps,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                logging.info(
                    "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(
                        save_path, epoch + 1, steps, time.time() - t1
                    )
                )

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": steps,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(
                    save_path, epoch + 1, steps, time.time() - t1
                )
            )


if __name__ == "__main__":
    main()
