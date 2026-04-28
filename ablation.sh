output_root="./Ablation"

# echo "Ablation on Covariance"
# python train_hsi.py \
#     --output_root ${output_root}/covariance \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 \
#     --covariance_type cholesky


# echo "Ablation on Prune"
# python train_hsi.py \
#     --output_root ${output_root}/no_prune \
#     --dataset all \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 \
#     --no_prune


# echo "Ablation on Addition"
# python train_hsi.py \
#     --output_root ${output_root}/no_add \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 10000 \
#     --num_gabor 3 \
#     --no_adaptive_add


# echo "Ablation on Addition and Prune"
# python train_hsi.py \
#     --output_root ${output_root}/no_add_no_prune \
#     --dataset all \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 10000 \
#     --num_gabor 3 \
#     --no_adaptive_add \
#     --no_prune


# echo "Ablation on Gabor"
# python train_hsi.py \
#     --output_root ${output_root}/no_gabor \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 0


# echo "Ablation on delta endmember"
# python train_hsi.py \
#     --output_root ${output_root}/freeze_endmember \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 \
#     --freeze_endmember


# echo "Ablation on SLV"
# python train_hsi.py \
#     --output_root ${output_root}/no_slv \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 \
#     --no_SLV_init



# echo "Ablation on NMF (skip A/E decomposition; render C channels directly)"
# python train_hsi.py \
#     --output_root ${output_root}/no_nmf \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 \
#     --no_nmf

# echo "Ablation on ALL"
# python train_hsi.py \
#     --output_root ${output_root}/all_off \
#     --dataset all \
#     --rank 12 \
#     --iterations 30000 \
#     --num_points 10000 \
#     --num_gabor 0 \
#     --no_SLV_init \
#     --no_adaptive_add \
#     --covariance_type cholesky \
#     --freeze_endmember

# echo "Comparison"
# python train_hsi.py \
#     --output_root ${output_root}/comparision \
#     --dataset all \
#     --prune_iter 500 \
#     --rank 12 \
#     --lora_rank 4 \
#     --iterations 30000 \
#     --num_points 5000 \
#     --max_num_points 10000 \
#     --num_gabor 3 

echo "Ablation on NMF"
python train_hsi.py \
    --output_root ${output_root}/NMF \
    --dataset all \
    --prune_iter 500 \
    --iterations 30000 \
    --num_points 5000 \
    --max_num_points 10000 \
    --num_gabor 3 \
    --no_nmf
