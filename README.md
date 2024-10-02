# Image Watermarking with PRC

PRC watermark is a method similar to Tree-Ring Watermark, where a diffusion model generates images embedded with a watermark as defined by a specified watermark key.

The output of the watermark detection is binary, indicating whether the watermark is detected or not.

We will also add how to embed and decode longer messages with PRC watermark later.

## Dependencies

The code is based on `python 3.11.9` and the packages specified in `requirements.txt`.

You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

You need to specify the number of test images to generate and test on. The example uses 10. The watermark key is randomly generated and saved in the `keys` folder.

```bash
python encode.py --test_num 10
```

```bash
python decode.py --test_num 10 --test_path [path to test images]
```

You can also change the model and prompt in `model_id` and `dataset_id` respectively.

Additionally, you can set the targeted False Positive Rate (FPR) using the `fpr` parameter. The default value is 0.00001.


**Note**: Need to change the huggingface cache directory in `encode.py` and `decode.py`.

## References

- [Treering Watermark](https://github.com/YuxinWenRick/tree-ring-watermark)
- [WAVES](https://github.com/umd-huang-lab/WAVES)
- [WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker)
- [Exact Inversion](https://github.com/smhongok/inv-dpm)
- [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading)