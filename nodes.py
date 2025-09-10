import logging
from typing import Any


class DiffusionSDXLFrameByFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Multiple Images"}),
                "model": ("MODEL", {"tooltip": "IC-Light model"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": {
                "start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Frame to start at.",
                    },
                ),
                "stop": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Frame to stop at.\nLeave at 0 to use all frames.",
                    },
                ),
                "step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "How much frames to step over each iteration.",
                    },
                ),
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "main"
    CATEGORY = "conditioning"
    DESCRIPTION = """Applies Diffusion SDXL for each input image individually and outputs the processed images.\n\nVersion: 0.0.2"""

    def main(
        self,
        images,  # torch.Tensor
        model,
        positive,
        negative,
        vae,
        start: int = 1,
        stop: int = 0,
        step: int = 1,
        cfg=8.0,
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

        start = max(1, int(start))
        step = max(1, int(step))

        # stop: 0 => use full length; otherwise inclusive index
        if stop <= 0 or stop >= total:
            stop = total
        else:
            stop = min(int(stop) + 1, total)  # inclusive

        if start >= total:
            return ([],)

        images = images[start:stop:step].contiguous()

        # * ENCODE
        try:
            # torch.Tensor
            encoded: Any = vae.encode(images)
        except Exception as e:
            logging.error(f"Error encoding images: {e}")
            return ([],)

        # samples is a tensor
        if encoded is None or encoded.numel() == 0:
            logging.error(f"Could not get samples from encoded images.")
            return ([],)

        decoded_batches = []

        for index, latent in enumerate(encoded):
            # each image latent is a tensor
            logging.info(f"Frame {index + 1}/{len(images)}")

        return (images,)


NODE_CLASS_MAPPINGS = {
    "DiffusionSDXLFrameByFrame": DiffusionSDXLFrameByFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionSDXLFrameByFrame": "Diffusion SDXL (Frame by Frame)",
}
