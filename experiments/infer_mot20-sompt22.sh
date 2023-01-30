python3 -W ignore ./src/demo.py mot \
	    --load_model /home/ubuntu/phd/FairCenterMOT/exp/mot/mot20-sompt22/logs_2023-01-28-06-22-fairmotx-mot20-sompt22/model_79.pth \
	        --input_video /home/ubuntu/phd/FairCenterMOT/src/data/SOMPT22/images/test/SOMPT22-14/img1 \
		    --reid_dim 128 --yolo_depth 0.67 --yolo_width 0.75 --reid_cls_id 0 \
		        --data_cfg /home/ubuntu/phd/FairCenterMOT/src/lib/cfg/mot20-sompt22.json \
			    --gpus 0
