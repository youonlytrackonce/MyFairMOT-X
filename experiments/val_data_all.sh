python ./src/track.py mot \
--exp_id data_all \
--load_model /home/fatih/phd/model_zoo/tracker/FairMOT-X/data_all/model_30.pth \
--gpus 0 \
--data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/data_all.json \
--reid_cls_id 0 --val_mot17 True
