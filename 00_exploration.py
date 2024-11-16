import argparse
import timeit

from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
    get_scheduler,
)

from data import RawTokenDataset, get_maskgit_collator
from visualize import decode_latents_wrapper, export_to_gif
from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT

def parse_args(arguments):
    # parser = argparse.ArgumentParser(description="Train a MaskGIT or Llama-style LLM on video generation.")
    parser = argparse.ArgumentParser(description="Train a spatial-temporal MaskGIT-style model on video generation.")

    # Data
    parser.add_argument(
        "--train_data_dir", type=str, default="data/train_v0.1",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v0.1",
        help="Directory containing tokenized data, should have a `video.bin`, `metadata.json` and `segment_ids.json`."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=15,
        help="Number of frames to in a sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Difference in frame count between consecutive frames in a sequence.",
    )
    parser.add_argument(
        "--filter_overlaps",
        action="store_true",
        help=(
            "Whether to filter repeated frames in the train dataset (`filter_overlaps` always true for the val set). "
            "Filtering essentially makes the training dataset less correlated but ~16x smaller, "
            "see the `filter_overlaps` argument in `RawTokenDataset` for details.")
        ,
    )

    # Model
    parser.add_argument(
        "--llama_config",
        type=str,
        help="`transformers.LlamaConfig` json. "
             "E.g. https://huggingface.co/1x-technologies/Llama_1B_v0/blob/main/config.json",
    )
    parser.add_argument(
        "--genie_config",
        type=str,
        help="GenieConfig json."
    ),
    parser.add_argument(
        "--warmstart_path",
        type=str,
        default=None,
        help="A path to a checkpoint to warmstart a model from, possibly not trained on the same dataset, "
             "will resize embeddings if needed.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # Training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=int(1e10),
        help="Only evaluate on `max_eval_steps` batches of validation data per process, faster.",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=1000,
        help="Eval every N training steps.",
    )
    parser.add_argument(
        "--vis_every_n_steps",
        type=int,
        default=1000,
        help="Visualize every N training steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "custom_cosine"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Threshold to clip gradients.",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.0,
        help="Attention dropout prob.",
    )
    parser.add_argument(
        "--adam_beta_1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta_2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
    )

    # Misc
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the model checkpoints.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="1000",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--overfit_first_batch",
        action="store_true",
        help=(
            "Debug option that trains and validates on only the first batch of the training dataset."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--mu_transfer",
        action="store_true",
        help="If specified, will train with mu transfer reparametrizations. Only supports Llama models."
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="If specified, will not compile the model."
    )

    args = parser.parse_args(arguments)

    return args



args = parse_args(["--train_data_dir", "data/train_v1.1/", "--val_data_dir", "data/val_v1.1/", "--genie_config", "genie/configs/magvit_n32_h8_d256.json", "--output_dir", "data/genie_model", "--max_eval_steps", "10"])

train_dataset = RawTokenDataset(args.train_data_dir, window_size=args.window_size,
                                stride=args.stride, filter_overlaps=args.filter_overlaps)

latent_side_len, vocab_size, hz = [train_dataset.metadata[key] for key in ("s", "vocab_size", "hz")]

config = GenieConfig.from_pretrained(args.genie_config)
config.use_mup = args.mu_transfer  # Note: changing this may affect pre-trained model due to attn scaling
config.image_vocab_size = vocab_size
config.T = args.window_size
config.S = latent_side_len**2

model = STMaskGIT(config)

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params/1e6, "M")

collate_fn = default_data_collator if args.llama_config is not None else get_maskgit_collator(config)
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn,
    batch_size=1,# args.per_device_train_batch_size,
    num_workers=4,
    pin_memory=True,
)

def step():
    model.train()
    model.to("cuda")
    batch = next(iter(train_dataloader))
    for key in batch:
        batch[key] = batch[key].to("cuda")
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    return loss, outputs.logits

print("Starting step")
t0 = timeit.default_timer()
loss, logits = step()
print("Loss:", loss.item())
print("Logits shape:", logits.shape)
print("Time:", timeit.default_timer() - t0)
print("Done")
