CUDA_VISIBLE_DEVICES=0 python loaddata.py -d market -a resnet50 -b 50 -j 4 --epochs 81 --log logs/humpbackWhale/ --combine-trainval --feature 256 --height 256 --width 256 --step-size 30 --data-dir ~/HumpbackWhale/dataset/
 
