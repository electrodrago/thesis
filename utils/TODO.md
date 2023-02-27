- masked_loss for L1 Loss
- Change Pixel Loss to L1

/usr/bin/python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=27214 /usr/local/lib/python3.8/dist-packages/mmedit/.mim/tools/train.py /content/RealBasicVSR/thesis/configs_A100.py --launcher pytorch