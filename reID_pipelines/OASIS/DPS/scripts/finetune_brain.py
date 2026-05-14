## finetune_brain.py

"""
Fine-tune ffhq_10m.pt on brain MRI slices.
Freezes input_blocks and middle_block, trains only output_blocks and out layers.
"""

import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.unet import create_model
from guided_diffusion.train_util import TrainLoop


# ── Dataset ──────────────────────────────────────────────────────────────────
class BrainMRIFinetuneDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path)
        self.images = data
        print(f"Loaded {len(self.images)} slices (subsampled from {len(data)}) from {npy_path}")
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]                   # (248, 248) float32 [0,1]

        # Resize to 256x256
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,248,248)
        img = torch.nn.functional.interpolate(
            img, size=(256, 256), mode='bilinear', align_corners=False)
        img = img.squeeze(0).repeat(3, 1, 1)     # (3,256,256)

        # Rescale from [0,1] to [-1,1] — required by diffusion training
        img = img * 2.0 - 1.0

        return img, {}                           # empty dict for cond


def save_best(model, save_dir, step, best_loss, loss_val, save_every=50):
    """Save only if improved AND at a save interval to reduce CPU/GPU transfers."""
    if loss_val < best_loss and step % save_every == 0:
        path = os.path.join(save_dir, 'best_model.pt')
        torch.save(model.state_dict(), path)
        logger.log(f"Saved best model at step {step} -> {path}")
        return loss_val
    return best_loss
    
def cycle_dataloader(dataloader):
    """Infinitely cycle through dataloader — required by TrainLoop."""
    while True:
        for batch in dataloader:
            yield batch


# ── Freeze logic ─────────────────────────────────────────────────────────────
def freeze_encoder(model):
    """Train output_blocks 6-11 and final out layer — ~15% of parameters."""
    frozen, trainable = 0, 0

    for name, param in model.named_parameters():
        should_train = (
            name.startswith('output_blocks.6') or
            name.startswith('output_blocks.7') or
            name.startswith('output_blocks.8') or
            name.startswith('output_blocks.9') or
            name.startswith('output_blocks.10') or
            name.startswith('output_blocks.11') or
            name.startswith('out.')
        )
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
        else:
            frozen += param.numel()

    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable fraction:   {trainable/(frozen+trainable)*100:.1f}%")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    # Replace the entire model+diffusion creation block with this:
    logger.log("Creating model and diffusion...")

    # Use DPS's create_model — correct architecture for ffhq_10m.pt
    from guided_diffusion.unet import create_model
    model = create_model(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    # Disable gradient checkpointing in all submodules
    for module in model.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False

    # Use script_util's create_gaussian_diffusion for the diffusion process
    from guided_diffusion.script_util import create_gaussian_diffusion
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=False,        # disable learned sigma — removes vb loss
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )

    # Load pretrained weights, skipping mismatched layers
    logger.log(f"Loading pretrained weights from {args.pretrained_ckpt}...")
    state_dict = torch.load(args.pretrained_ckpt, map_location='cpu')

    # Remove keys that don't match current architecture
    model_state = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k not in filtered_state_dict]
    if skipped:
        logger.log(f"Skipped {len(skipped)} mismatched keys: {skipped}")

    model.load_state_dict(filtered_state_dict, strict=False)
    logger.log(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} layers from pretrained checkpoint.")


    # Freeze encoder
    model = freeze_encoder(model)
    model.to(dist_util.dev())

    # Dataloader
    logger.log("Creating dataloader...")
    dataset = BrainMRIFinetuneDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    data = cycle_dataloader(dataloader)

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    logger.log("Starting fine-tuning...")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    best_loss = float('inf')

    for step in range(args.lr_anneal_steps):
        model.train()
        batch, _ = next(data)
        batch = batch.to(dist_util.dev())

        t, weights = schedule_sampler.sample(batch.shape[0], dist_util.dev())

        losses = diffusion.training_losses(model, batch, t)
        loss = (losses['loss'] * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            max_norm=1.0
        )
        optimizer.step()

        loss_val = loss.item()

        # Save best model every step — always overwrites best_model.pt
        best_loss = save_best(model, args.save_dir, step, best_loss, loss_val, save_every=50)

        if step % args.log_interval == 0:
            logger.log(f"step={step} loss={loss_val:.4f} best={best_loss:.4f}")

    logger.log(f"Training complete. Best loss: {best_loss:.4f}")
    logger.log(f"Best model saved at: {os.path.join(args.save_dir, 'best_model.pt')}")


def create_argparser():
    defaults = dict(
            data_path="./data/oasis_finetune.npy",
            pretrained_ckpt="./models/ffhq_10m.pt",
            save_dir="./checkpoints/brain_finetune",
            schedule_sampler="uniform",
            lr=1e-5,
            weight_decay=0.0,
            lr_anneal_steps=2000,      # reduced from 50000
            batch_size=4,
            microbatch=-1,
            ema_rate="0.9999",
            log_interval=100,
            save_interval=2000,         # save every 2000 steps
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
        )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    return parser


if __name__ == "__main__":
    main()
