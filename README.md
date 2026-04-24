# LoR-SGS: Hyperspectral Image Compression via Low-rank Spectral Gaussian Splatting
### Requirements

```bash
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

### Setup

Organize your files as follows:

```kotlin
HSI/
 ├── data/
 │    └── PaviaU.mat
 └── init/
```

The `.mat` file contains hyperspectral image data.

The estimated coefficient basis matrix file will be automatically generated in `HSI/init/`.

The HSI fitting entrypoint is [train_hsi.py](train_hsi.py). The implementation keeps the cholesky projection path, Gabor rasterization, and LoRA fine-tuning of the endmember matrix E, while leaving out density control.

### Run demo

Run the following command to perform HSI fitting with Gabor splatting and LoRA endmember tuning on the *JasperRidge* dataset:

```shell
python train_hsi.py --dataset jasperridge --rank 10 --num_points 600 --iterations 8000 --num_gabor 2 --lora_rank 2 --lora_alpha 0.1
```

## Acknowledgments

This implementation is developed based on the open-source project [GaussianImage](https://github.com/Xinjie-Q/GaussianImage), which provides the foundation for Gaussian splatting. We have modified and extended it for low-rank spectral modeling and hyperspectral image compression. We thank the original authors for their excellent work and for sharing their code.

## Citation

If you use our method or our code in your research, please kindly cite it:

```latex
@article{wang2025lorsgs,
  title={LoR-SGS: Hyperspectral Image Compression via Low-Rank Spectral Gaussian Splatting},
  author={Li, Tianyu and Wang, Ting and Zhao, Xile and Wang, Chao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

