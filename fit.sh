# for rank in 8 10 12 14
# do
#     for lora_rank in 3 4
#     do
#         for max_num_points in 5000 10000 20000 30000
#         do
#             for num_gabor in 1 2 3
#             do
#                 num_points=$((max_num_points / 2))

#                 echo "Running: rank=$rank lora_rank=$lora_rank max_points=$max_num_points num_points=$num_points num_gabor=$num_gabor"

#                 python train_hsi.py \
#                     --dataset all \
#                     --prune_iter 500 \
#                     --rank $rank \
#                     --lora_rank $lora_rank \
#                     --iterations 30000 \
#                     --num_points $num_points \
#                     --max_num_points $max_num_points \
#                     --num_gabor $num_gabor
#             done
#         done
#     done
# done


python train_hsi.py --dataset Urban --rank 12 --lora_rank 4 --max_num_points 10000 --num_points 5000 --num_gabor 3  --prune_iter 500 