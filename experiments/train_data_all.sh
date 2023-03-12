python ./src/train.py mot \
--exp_id data_all --yolo_depth 0.67 --yolo_width 0.75 \
--lr 1e-3 --lr_step 2 \
--reid_dim 128 --augment --mosaic \
--load_model /home/fatih/phd/FairCenterMOT/exp/mot/data_all/logs_2023-03-07-12-15/model_last.pth \
--batch_size 12 --gpus 0 \
--data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/data_all.json \
--reid_cls_id 0 --num_epochs 150 --num_workers 8 --resume --start_epoch 31
