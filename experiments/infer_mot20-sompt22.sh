python3 -W ignore ./src/demo.py mot \
	    --load_model /home/fatih/phd/FairCenterMOT/exp/mot/data_all/model_last.pth \
	        --input_video /home/fatih/phd/FairCenterMOT/src/data/SOMPT22/images/test/SOMPT22-09/img1 \
		    --reid_dim 128 --yolo_depth 0.67 --yolo_width 0.75 --reid_cls_id 0 \
		        --data_cfg /home/fatih/phd/FairCenterMOT/src/lib/cfg/data_all.json \
			    --gpus 0
