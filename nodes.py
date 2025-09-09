import logging


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
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
            },
            "optional": {
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "Plug in a latent image for the sampler, otherwise an empty latent is used."
                    },
                ),
                "opt_background": ("LATENT",),
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
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 0.18215,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Conditioning Multiplier",
                    },
                ),
                "add_noise": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Add noise to sampler."},
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Sampling Noise Seed",
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
    CATEGORY = "IC-Light"
    DESCRIPTION = """Applies IC-Light to each images of images input. Encodes, conditions, samples and decodes them.\n\nPlug in a latent image for the sampler, otherwise an empty latent is used.\n\nVersion: 0.0.9"""

    def main(
        self,
        images,  # torch.Tensor
        model,
        positive,
        negative,
        vae,
        sampler,
        sigmas,
        latent_image=None,
        opt_background=None,
        start: int = 1,
        stop: int = 0,
        step: int = 1,
        multiplier=0.18215,
        add_noise=True,
        noise_seed=0,
        cfg=8.0,
    ):

        logging.info("------------------")
        logging.info("| Diffusion SDXL (Frame by Frame) |")
        logging.info("------------------")

        return (images,)


NODE_CLASS_MAPPINGS = {
    "Diffusion SDXL (Frame by Frame)": DiffusionSDXLFrameByFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusion SDXL (Frame by Frame)": "IC-Light Video (Frame by Frame)",
}
