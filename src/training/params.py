import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, default=None,
        help="Path to the LMDB directory with training data split",
    )
    parser.add_argument(
        "--val-data", type=str, default=None,
        help="Path to the LMDB directory with validation data split, default to None which disables validation",
    )
    parser.add_argument(
        "--emb-path", type=str, default=None,
        help="item embedding lmdb dir",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="The number of workers for dataloader."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name", type=str, default="train_clip",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="How often to log loss info."
    )
    parser.add_argument(
        "--report-training-batch-acc", default=False, action="store_true",
        help="Whether to report training batch accuracy."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--valid-user-batch-size", type=int, default=64,
        help="Batch size for validation per GPU."
    )
    parser.add_argument(
        "--valid-item-batch-size", type=int, default=64,
        help="Batch size for validation per GPU."
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Number of steps to train for (in higher priority to --max_epochs)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=32,
        help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--valid-step-interval", type=int, default=None,
        help="The step interval for validation (default to None which disables validation between steps)."
    )
    parser.add_argument(
        "--valid-epoch-interval", type=int, default=1,
        help="The epoch interval for validation (default to 1, set None to disable validation between epochs)."
    )

    parser.add_argument(
        "--max-one-year-length", type=int, default=256,
    )
    parser.add_argument(
        "--max-one-month-length", type=int, default=20,
    )
    parser.add_argument(
        "--sample-one-year-length", type=int, default=256,
    )
    parser.add_argument(
        "--sample-one-month-length", type=int, default=20
    )

    parser.add_argument(
        "--num-layers", type=int, default=12,
        help="The layer number of the bert model."
    )
    parser.add_argument(
        "--input-dim", type=int, default=4096,
        help="The layer number of the bert model."
    )
    parser.add_argument(
        "--output-dim", type=int, default=64,
        help="The layer number of the bert model."
    )

    parser.add_argument(
        "--train-stage", type=str, default='pretrain', choices=['pretrain', 'finetune'],
        help="The stage of training."
    )
    parser.add_argument(
        "--ft-data", type=str, default='Scientific',
        choices=['Pet', 'Games', 'Instruments', 'Arts', 'Office', 'Scientific'], help="The stage of training."
    )

    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup", type=float, default=500, help="Number of steps to warmup for.")

    # amazon books
    parser.add_argument("--test-user-data", default=None, type=str)
    parser.add_argument("--test-item-data", default=None, type=str)
    parser.add_argument("--test-emb-lmdb", default=None, type=str)

    parser.add_argument("--use-bn-sync", default=False, action="store_true",
                        help="Whether to use batch norm sync.")

    parser.add_argument("--skip-scheduler", action="store_true", default=False,
                        help="Use this flag to skip the learning rate decay.",
                        )
    parser.add_argument("--save-epoch-frequency", type=int, default=1,
                        help="How often to save checkpoints by epochs."
                        )
    parser.add_argument("--save-step-frequency", type=int, default=-1,
                        help="How often to save checkpoints by steps."
                        )
    parser.add_argument("--resume", default=None, type=str,
                        help="path to latest checkpoint (default: none)",
                        )
    parser.add_argument("--reset-optimizer", action="store_true", default=False,
                        help="If resumed from a checkpoint, whether to reset the optimizer states.",
                        )
    parser.add_argument("--precision", choices=["amp", "fp16", "bf16", "fp32"], default="amp",
                        help="Floating point precision."
                        )

    parser.add_argument("--skip-aggregate", default=False, action="store_true",
                        help="whether to aggregate features across gpus before computing the loss"
                        )
    parser.add_argument("--debug", default=False, action="store_true",
                        help="If true, more information is logged."
                        )
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed."
                        )

    parser.add_argument("--grad-checkpointing", default=False, action='store_true',
                        help="Enable gradient checkpointing.",
                        )
    parser.add_argument("--accum-freq", type=int, default=1,
                        help="Update the model every --acum-freq steps."
                        )

    parser.add_argument("--wandb_project", default=None,
                        help="wandb project name",
                        )

    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    return args
