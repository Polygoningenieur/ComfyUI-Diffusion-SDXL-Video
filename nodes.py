import torch
import logging
import traceback
from typing import Any
from nodes import VAEEncode, ControlNetApplyAdvanced, KSampler


class DiffusionSDXLFrameByFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Multiple Images"}),
                "model": ("MODEL", {"tooltip": "IC-Light model"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "vae": ("VAE",),
            },
            "optional": {
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Sampling cfg",
                    },
                ),
                "controlnet_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "ControlNet Strength",
                    },
                ),
                "controlnet_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "ControlNet Start Percent",
                    },
                ),
                "controlnet_end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "ControlNet End Percent",
                    },
                ),
                "frame_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Frame to start at.",
                    },
                ),
                "frame_stop": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Frame to stop at.\nLeave at 0 to use all frames.",
                    },
                ),
                "frame_step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "How much frames to step over each iteration.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "main"
    CATEGORY = "conditioning"
    DESCRIPTION = """Applies Diffusion SDXL for each input image individually and outputs the processed images.\n\nVersion: 0.0.4"""

    def main(
        self,
        images,  # torch.Tensor
        model,
        positive,
        negative,
        control_net,
        vae,
        cfg=8.0,
        controlnet_strength: float = 1.0,
        controlnet_start_percent: float = 0.0,
        controlnet_end_percent: float = 1.0,
        frame_start: int = 1,
        frame_stop: int = 0,
        frame_step: int = 1,
    ):

        logging.info("-----------------------------------")
        logging.info("| Diffusion SDXL (Frame by Frame) |")
        logging.info("-----------------------------------")

        # cut and slice images to provided start, stop and step
        total = (
            int(images.shape[0]) if hasattr(images, "shape") and images.ndim >= 1 else 0
        )
        if total == 0:
            return ([],)

        frame_start = max(1, int(frame_start))
        frame_step = max(1, int(frame_step))

        # stop: 0 => use full length; otherwise inclusive index
        if frame_stop <= 0 or frame_stop >= total:
            frame_stop = total
        else:
            frame_stop = min(int(frame_stop) + 1, total)  # inclusive

        if frame_start >= total:
            return ([],)

        images = images[frame_start:frame_stop:frame_step].contiguous()

        # * ENCODE
        try:
            # ({"samples": tensor})
            encoded: tuple[dict[str, Any]] = VAEEncode.encode(self, vae, images)
        except Exception as e:
            logging.error(f"Error encoding images: {e}")
            return ([],)

        # samples is a tensor
        samples_tensor: Any = encoded[0].get("samples", None)
        if samples_tensor is None or samples_tensor.numel() == 0:
            logging.error(f"Could not get samples from encoded images.")
            return ([],)

        decoded_batches = []

        for index in range(samples_tensor.shape[0]):
            latent = samples_tensor[index].unsqueeze(0)  # shape [1, 4, 64, 64]
            # each image latent is a tensor
            logging.info(f"Frame {index + 1}/{len(images)}")

            # * CONTROL NET
            controlnet_positive: list = None
            controlnet_negative: list = None
            try:
                (controlnet_positive, controlnet_negative) = (
                    ControlNetApplyAdvanced.apply_controlnet(
                        self,
                        positive=positive,
                        negative=negative,
                        control_net=control_net,
                        image=latent.squeeze(0),
                        strength=controlnet_strength,
                        start_percent=controlnet_start_percent,
                        end_percent=controlnet_end_percent,
                        vae=vae,
                        extra_concat=[],
                    )
                )
            except Exception as e:
                logging.error(f"Error conditioning latent image: {e}")
                continue
            if None in [controlnet_positive, controlnet_negative]:
                continue

            # * SAMPLING
            sampled_latent: dict[str, Any] = None  # {"samples": tensor}
            latent_dict = {"samples": latent}
            try:
                sampled_latent = KSampler.sample(
                    self,
                    model=model,
                    seed=0,  # provide a valid seed
                    steps=20,  # provide valid steps
                    cfg=cfg,
                    sampler_name="euler",  # provide valid sampler_name
                    scheduler="normal",  # provide valid scheduler
                    positive=controlnet_positive,
                    negative=controlnet_negative,
                    latent_image=latent_dict,
                    denoise=1.0,
                )
            except Exception as e:
                logging.error(f"Error sampling image: {e}")
                # logging.error(traceback.format_exc())
                continue
            if sampled_latent[0] is None:
                continue

            # decode sampled latent images -> tensor [B,H,W,C]
            try:
                # TODO change to KSampler sampled latent
                decoded = vae.decode(sampled_latent[0])
                # Flatten any extra batch/temporal dims to [N,H,W,C]
                if hasattr(decoded, "shape") and len(decoded.shape) == 5:
                    decoded = decoded.reshape(
                        -1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1]
                    )
            except Exception as e:
                logging.error(f"Error decoding sampled image: {e}")
                continue

            decoded_batches.append(decoded)

        # Concatenate all frames into a single IMAGE tensor [N,H,W,C]
        if len(decoded_batches) == 0:
            return (images,)
        frames_out = torch.cat(decoded_batches, dim=0)
        return (frames_out,)


NODE_CLASS_MAPPINGS = {
    "DiffusionSDXLFrameByFrame": DiffusionSDXLFrameByFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionSDXLFrameByFrame": "Diffusion SDXL (Frame by Frame)",
}
