python3 -W ignore ./src/demo.py mot \
	    --load_model /home/fatih/phd/FairCenterMOT/exp/mot/data_all/logs_2023-03-15-16-31/model_last.pth \
	        --input_video /home/fatih/phd/FairCenterMOT/src/data/MOT17/images/train/MOT17-13-DPM/img1 \
		    --reid_dim 128 --yolo_depth 0.67 --yolo_width 0.75 --reid_cls_id 0 \
		        --data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/data_all.json \
			    --gpus 0
