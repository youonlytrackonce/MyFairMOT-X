python ./src/train.py mot \
--exp_id mot20-sompt22 --yolo_depth 0.67 --yolo_width 0.75 \
--lr 1e-3 --lr_step 2 \
--reid_dim 128 --augment --mosaic \
--load_model /home/fatih/phd/FairCenterMOT/exp/mot/mot20-sompt22/logs_2023-01-28-06-22-fairmotx-mot20-sompt22/model_79.pth \
--batch_size 4 --gpus 0 --resume \
--data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/mot20-sompt22.json \
--reid_cls_id 0 --num_epochs 150 --num_workers 8
