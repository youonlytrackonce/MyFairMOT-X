python ./src/train.py mot \
--exp_id data_all --yolo_depth 0.67 --yolo_width 0.75 \
--lr 1e-3 --lr_step 2 \
--reid_dim 128 --augment --mosaic \
--load_model /home/fatih/phd/FairCenterMOT/models/yolox_s.pth \
--batch_size 16 --gpus 0 \
--data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/data_all.json \
--reid_cls_id 0 --num_epochs 30 --num_workers 8
