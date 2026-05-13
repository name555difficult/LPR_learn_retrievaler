```shell
CUDA_VISIBLE_DEVICES=3 nohup python train_retrievaler.py --config ../config/config_oxford_retrievaler.txt --model_config ../models/hotformerloc_oxford_cfg.txt --pretrained_weights /mnt/data16t-A/wzb/LPR_learn_retrievaler/weights/offical_weights/hotformerloc_oxford.pth > training_base.log 2<&1 &

```