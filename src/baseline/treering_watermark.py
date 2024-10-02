import numpy as np
import torch
from torchvision import transforms
import PIL
import pickle
from diffusers import DDIMInverseScheduler
from typing import Union, List, Tuple
import hashlib
import os


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def _get_pattern(shape, w_pattern='ring', generator=None):
    gt_init = torch.randn(shape, generator=generator)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = gt_patch.clone().detach()
        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str) -> torch.Tensor:
def tr_get_noise(shape: Union[torch.Size, List, Tuple], keys_path, from_file: str = None, generator=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not from_file:
        # for now we hard code all hyperparameters
        w_channel = 0  # id for watermarked channel
        w_radius = 10  # watermark radius
        w_pattern = 'rand'  # watermark pattern

        # get watermark key and mask
        np_mask = _circle_mask(shape[-1], r=w_radius)
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, w_channel] = torch_mask

        w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator)

        # inject watermark
        assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
        assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

        init_latents = torch.randn(shape, generator=generator)

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_fft[w_mask] = w_key[w_mask].clone()
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

        # convert the tensor to bytes
        tensor_bytes = init_latents.numpy().tobytes()

        # generate a hash from the bytes
        hash_object = hashlib.sha256(tensor_bytes)
        hex_dig = hash_object.hexdigest()

        file_name = "_".join([hex_dig, str(w_channel), str(w_radius), w_pattern]) + ".pkl"
        file_path = os.path.join(keys_path, file_name)
        print(f"Saving watermark key to {file_path}")
        with open(f'{file_path}', 'wb') as f:
            pickle.dump((w_key, w_mask), f)

    else:
        file_name = f"{from_file}.pkl"
        file_path = os.path.join(keys_path, file_name)

        with open(f'{file_path}', 'rb') as f:
            w_key, w_mask = pickle.load(f)
        init_latents = torch.randn(shape, generator=generator)

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_fft[w_mask] = w_key[w_mask].clone()
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_mask


def _transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def tr_detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe, keys_path, model_hash):
    detection_time_num_inference = 50
    threshold = 72

    file_name = f"{model_hash}.pkl"
    file_path = os.path.join(keys_path, file_name)

    with open(f'{file_path}', 'rb') as f:
        w_key, w_mask = pickle.load(f)

    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    inverted_latents = pipe(
        prompt='',
        latents=image_latents,
        guidance_scale=1,
        num_inference_steps=detection_time_num_inference,
        output_type='latent',
    )
    inverted_latents = inverted_latents.images.float().cpu()

    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

    if dist <= threshold:
        pipe.scheduler = curr_scheduler
        return dist, True

    return dist, False
