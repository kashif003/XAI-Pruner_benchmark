# this file will contain the notes and task that needed to be done to replicate this repo.

## 2. Code changes
- `prune-CNN.py:294` -> `momentum=args.pruning_momentum, num_classes=args.num_classes)`
- `prune-CNN.py:194` -> `folder_name="subImageNet"`
- `prune-ViT.py:150` and `prune-CNN.py:147` -> add `'EVO_IMNET'` to `choices`
- `prune-ViT.py:194` and `prune-CNN.py:191` -> add below: `import random; random.seed(seed)`

## Dataset path
Results use ImageNet-2012 only. Set your path in these defaults (marked `# CHANGE:`), or pass it as a flag, which overrides them:
- `prune-ViT.py:149` (`--data_path`)
- `prune-CNN.py:146` (`--data_path`)
- `fintune.py:155` (`--data_path`)
- `lib/subImageNet.py:9` (`--data-path`)

Your folder needs `train/`, `val/`, `subImageNet/`. No spaces in the path. Replace `/data/imagenet/` in the commands below.

## 3. Run (change only --model, --resume, --pruning_rate, output folders)
Prune DeiT:
`python prune-ViT.py --data_set EVO_IMNET --data_path /data/imagenet/ --imagenet_default_mean_and_std True --model deit_tiny_patch16_224 --resume weights/deit_tiny_patch16_224-a1311bcf.pth --output_dir out/tiny --batch-size 128 --pruning_rate 0.246 --pruning_epsilon 0.02`

Prune ResNet-50:
`python prune-CNN.py --data_set EVO_IMNET --data_path /data/imagenet/ --imagenet_default_mean_and_std True --model resnet50 --resume weights/resnet50-0676ba61.pth --output_dir out/r50 --batch-size 128 --pruning_rate 0.40`

Fine-tune (8 GPUs; for fewer, set `--lr` = 1e-3 x total batch / 1024):
`python -m torch.distributed.launch --nproc_per_node=8 --use_env fintune.py --model deit_tiny_patch16_224 --data_path /data/imagenet/ --imagenet_default_mean_and_std True --cfg out/tiny/state.yaml --resume out/tiny/checkpoint_pruned.pth --output_dir out/tiny_ft --epochs 300 --batch-size 128 --warmup-epochs 0 --cooldown-epochs 0`

## 4. Targets (check size after pruning, before fine-tuning)
| Model | --pruning_rate | Params | GFLOPs | Top-1 |
|---|---|---|---|---|
| DeiT-Tiny | 0.246 | 4.2M | 0.9 | 70.99 |
| DeiT-Small | 0.315 | 14.7M | 3.1 | 80.33 |
| DeiT-Small | 0.498 | 10.8M | 2.3 | 77.98 |
| ResNet-50 | start 0.40, adjust | 13.21M | 1.87 | 76.10 |

Accuracy per epoch is in `log.txt` in each output folder. Run Tiny first.

## 5. Unknowns
- Exact match unlikely: pruned heads/channels, `filter_percent`, `protect_rate` not published.
- No ResNet-50 fine-tuning settings published; start with the defaults.
- Fine-tuning not tested yet (needs GPU).
