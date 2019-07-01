CUDA_VISIBLE_DEVICES=0 python3 predict.py
kaggle competitions submit -c imaterialist-fashion-2019-FGVC6 -f submission.csv -m "model8 final"
