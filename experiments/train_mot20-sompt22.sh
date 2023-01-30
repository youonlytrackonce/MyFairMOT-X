python ./src/train.py mot \
--exp_id mot20-sompt22 --yolo_depth 0.67 --yolo_width 0.75 \
--lr 7e-4 --lr_step 2 \
--reid_dim 128 --augment --mosaic \
--load_model /home/ubuntu/phd/FairCenterMOT/models/yolox_s.pth \
--batch_size 1 --gpus 0 \
--data_cfg /home/ubuntu/phd/FairCenterMOT/src/lib/cfg/mot20-sompt22.json \
--reid_cls_id 0 --num_epochs 1 --num_workers 14
