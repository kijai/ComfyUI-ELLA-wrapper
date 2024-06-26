import os
from typing import Any, Optional, Union, Tuple
from contextlib import nullcontext
import safetensors.torch
import torch
try:
    from diffusers import (
        DPMSolverMultistepScheduler, 
        StableDiffusionPipeline, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        AutoencoderKL, 
        UNet2DConditionModel, 
        LCMScheduler, 
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler,
        UniPCMultistepScheduler
    )
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config,
        create_text_encoder_from_ldm_clip_checkpoint
    )            
except:
    raise ImportError("Diffusers version too old. Please update to 0.26.0 minimum.")
from .scheduling_tcd import TCDScheduler
from contextlib import nullcontext
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

from .hidiffusion import apply_hidiffusion, remove_hidiffusion

from omegaconf import OmegaConf
from .model import ELLA, T5TextEmbedder
from transformers import CLIPTokenizer
import comfy.model_management as mm
import comfy.utils
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class ELLAProxyUNet(torch.nn.Module):
    def __init__(self, ella, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ella
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

        self.flexible_max_length_workaround = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.flexible_max_length_workaround is not None:
            time_aware_encoder_hidden_state_list = []
            for i, max_length in enumerate(self.flexible_max_length_workaround):
                time_aware_encoder_hidden_state_list.append(
                    self.ella(encoder_hidden_states[i : i + 1, :max_length], timestep)
                )
            # No matter how many tokens are text features, the ella output must be 64 tokens.
            time_aware_encoder_hidden_states = torch.cat(
                time_aware_encoder_hidden_state_list, dim=0
            )
        else:
            time_aware_encoder_hidden_states = self.ella(
                encoder_hidden_states, timestep
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
    
class ella_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            
            },
        }

    RETURN_TYPES = ("ELLAMODEL",)
    RETURN_NAMES = ("ella_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "ELLA-Wrapper"

    def loadmodel(self, model, clip, vae):
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        device = mm.get_torch_device()

        custom_config = {
            'model': model,
            'vae': vae,
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(5)
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))

            print("loading ELLA")
            checkpoint_path = os.path.join(folder_paths.models_dir,'ella')
            ella_path = os.path.join(checkpoint_path, 'ella-sd1.5-tsc-t5xl.safetensors')
            if not os.path.exists(ella_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="QQGYLab/ELLA", local_dir=checkpoint_path, local_dir_use_symlinks=False)
            
            with (init_empty_weights() if is_accelerate_available() else nullcontext()):
                converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
                new_vae = AutoencoderKL(**converted_vae_config)

                converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
                unet = UNet2DConditionModel(**converted_unet_config)
                
            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)

            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            if is_accelerate_available():
                for key in converted_vae:
                    set_module_tensor_to_device(new_vae, key, device=device, dtype=dtype, value=converted_vae[key])
            else:
                new_vae.load_state_dict(converted_vae, strict=False)
            del converted_vae
            pbar.update(1)

            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            if is_accelerate_available():
                for key in converted_unet:
                    set_module_tensor_to_device(unet, key, device=device, dtype=dtype, value=converted_unet[key])
            else:
                unet.load_state_dict(converted_unet, strict=False)
            del converted_unet

            ella = ELLA()    
            safetensors.torch.load_model(ella, ella_path, strict=True)
            ella.to(device, dtype=dtype)
            ella_unet = ELLAProxyUNet(ella, unet)

            pbar.update(1)
            # 3. text_model
            print("loading text model")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1
            }
            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            scheduler=DPMSolverMultistepScheduler(**scheduler_config)
            pbar.update(1)
            del sd

            pbar.update(1)

            print("creating pipeline")
            self.pipe = StableDiffusionPipeline(
                unet=ella_unet,
                vae=new_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                image_encoder=None
            )
            print("pipeline created")
            pbar.update(1)
            self.pipe.enable_model_cpu_offload()
            ella_model = {
                'pipe': self.pipe,
            }
   
        return (ella_model,)
    
class diffusers_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            
            },
        }

    RETURN_TYPES = ("DIFFUSERSMODEL",)
    RETURN_NAMES = ("diffusers_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "ELLA-Wrapper"

    def loadmodel(self, model, clip, vae):
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        device = mm.get_torch_device()

        custom_config = {
            'model': model,
            'vae': vae,
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(5)
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))

            print("loading ELLA")
            checkpoint_path = os.path.join(folder_paths.models_dir,'ella')
            ella_path = os.path.join(checkpoint_path, 'ella-sd1.5-tsc-t5xl.safetensors')
            if not os.path.exists(ella_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="QQGYLab/ELLA", local_dir=checkpoint_path, local_dir_use_symlinks=False)
            
            with (init_empty_weights() if is_accelerate_available() else nullcontext()):
                converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
                new_vae = AutoencoderKL(**converted_vae_config)

                converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
                unet = UNet2DConditionModel(**converted_unet_config)
                
            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)

            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            if is_accelerate_available():
                for key in converted_vae:
                    set_module_tensor_to_device(new_vae, key, device=device, dtype=dtype, value=converted_vae[key])
            else:
                new_vae.load_state_dict(converted_vae, strict=False)
            del converted_vae
            pbar.update(1)

            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            if is_accelerate_available():
                for key in converted_unet:
                    set_module_tensor_to_device(unet, key, device=device, dtype=dtype, value=converted_unet[key])
            else:
                unet.load_state_dict(converted_unet, strict=False)
            del converted_unet

            pbar.update(1)
            # 3. text_model
            print("loading text model")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1
            }
            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            scheduler=DPMSolverMultistepScheduler(**scheduler_config)
            pbar.update(1)
            del sd

            pbar.update(1)

            print("creating pipeline")
            self.pipe = StableDiffusionPipeline(
                unet=unet,
                vae=new_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                image_encoder=None
            )
            print("pipeline created")
            pbar.update(1)
            #self.pipe.enable_model_cpu_offload()
            diffusers_model = {
                'pipe': self.pipe,
            }
   
        return (diffusers_model,)

class diffusers_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "diffusers_model": ("DIFFUSERSMODEL",),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 10.0, "min": 1.01, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler',
                    'UniPCMultistepScheduler',
                    'TCDScheduler'
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
            "prompt": ("STRING", {"default": "positive", "multiline": True}),
            "n_prompt": ("STRING", {"default": "negative", "multiline": True}),
            "hidiffusion": ("BOOLEAN", {"default": False}),
            },    
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "ELLA-Wrapper"

    def process(self, diffusers_model, width, height, steps, guidance_scale, seed, scheduler, prompt, n_prompt, hidiffusion):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        pipe=diffusers_model['pipe']
        pipe.to(device, dtype=dtype)

        scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1,
            }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == 'UniPCMultistepScheduler':
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        elif scheduler == 'TCDScheduler':
            noise_scheduler = TCDScheduler(**scheduler_config)
        
        pipe.scheduler = noise_scheduler
        if hidiffusion:
            apply_hidiffusion(pipe)
        else:
            remove_hidiffusion(pipe)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            

            images = pipe(
            prompt = prompt,
            negative_prompt = n_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=[
            torch.Generator(device=device).manual_seed(seed + i)
            for i in range(1)
            ],
                output_type="pt",
            ).images

            image_out = images.permute(0, 2, 3, 1).cpu().float()
            return (image_out,)
        
class ella_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ella_model": ("ELLAMODEL",),
            "ella_embeds": ("ELLAEMBEDS",),
            "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 10.0, "min": 1.01, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler',
                    'UniPCMultistepScheduler',
                    'TCDScheduler'
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
            },
            "optional": {
                "hidiffusion": ("BOOLEAN", {"default": False}),
            } 
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "ELLA-Wrapper"

    def process(self, ella_embeds, width, height, steps, guidance_scale, seed, ella_model, scheduler, hidiffusion=False):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        pipe=ella_model['pipe']
        pipe.to(device, dtype=dtype)

        scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1,
            }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == 'UniPCMultistepScheduler':
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        elif scheduler == 'TCDScheduler':
            noise_scheduler = TCDScheduler(**scheduler_config)
        
        pipe.scheduler = noise_scheduler
        if hidiffusion:
            apply_hidiffusion(pipe)
        else:
            remove_hidiffusion(pipe)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            
            # diffusers pipeline concatenate `prompt_embeds` too early...
            # https://github.com/huggingface/diffusers/blob/b6d7e31d10df675d86c6fe7838044712c6dca4e9/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L913
            pipe.unet.flexible_max_length_workaround = [ella_embeds["negative_prompt_embeds"].size(1)] * ella_embeds["batch_size"] + [ella_embeds["prompt_embeds"].size(1)] * ella_embeds["batch_size"]

            images = pipe(
            prompt_embeds=ella_embeds["prompt_embeds"],
            negative_prompt_embeds=ella_embeds["negative_prompt_embeds"],
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=[
            torch.Generator(device=device).manual_seed(seed + i)
            for i in range(ella_embeds["batch_size"])
            ],
                output_type="pt",
            ).images

            image_out = images.permute(0, 2, 3, 1).cpu().float()
            return (image_out,)


class ella_t5_embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "A vivid red book with a smooth, matte cover lies next to a glossy yellow vase. The vase, with a slightly curved silhouette, stands on a dark wood table with a noticeable grain pattern. The book appears slightly worn at the edges, suggesting frequent use, while the vase holds a fresh array of multicolored wildflowers.",}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            "max_length": ("INT", {"default": 128, "min": 1, "max": 512, "step": 1}),
            "fixed_negative": ("BOOLEAN", {"default": False}),
            },    
            "optional": {
            "flexible_max_length": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ELLAEMBEDS",)
    RETURN_NAMES = ("ella_embeds",)
    FUNCTION = "process"
    CATEGORY = "ELLA-Wrapper"

    def process(self, prompt, batch_size, max_length, fixed_negative, flexible_max_length=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()

        #old checkpoint location
        repo_id_old = "ybelkada/flan-t5-xl-sharded-bf16"
        checkpoint_path_old = os.path.join(script_directory, 'checkpoints')
        t5_path_old = os.path.join(checkpoint_path_old, repo_id_old)          

        if os.path.exists(t5_path_old):    
            t5_encoder = T5TextEmbedder(pretrained_path=t5_path_old).to(device, dtype=dtype)
        else:
            t5_path = os.path.join(folder_paths.models_dir,'t5_model', 'flan-t5-xl-encoder-only-bf16')
            if not os.path.exists(t5_path): 
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/flan-t5-xl-encoder-only-bf16", local_dir=t5_path, local_dir_use_symlinks=False)
            t5_encoder = T5TextEmbedder(pretrained_path=t5_path).to(device, dtype=dtype)

        t5_encoder.to(device)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            print("generating embeds")
            prompt = [prompt] * batch_size    
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if flexible_max_length:
                max_length = None
            prompt_embeds = t5_encoder(prompt, max_length=max_length).to(device, dtype)
            negative_prompt_embeds = t5_encoder([""] * batch_size, max_length=max_length if fixed_negative else None).to(device, dtype)

            max_length = max([prompt_embeds.size(1), negative_prompt_embeds.size(1)])
            b, _, d = prompt_embeds.shape
            prompt_embeds = torch.cat(
                [
                    prompt_embeds,
                    torch.zeros(
                        (b, max_length - prompt_embeds.size(1), d), device=device, dtype=dtype
                    ),
                ],
                dim=1,
            )
            negative_prompt_embeds = torch.cat(
                [
                    negative_prompt_embeds,
                    torch.zeros(
                        (b, max_length - negative_prompt_embeds.size(1), d),
                        device=device,
                        dtype=dtype,
                    ),
                ],
                dim=1,
            )
            embeds = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "batch_size": batch_size
            }
            t5_encoder.to(offload_device)
            return (embeds,)

NODE_CLASS_MAPPINGS = {
    "ella_model_loader": ella_model_loader,
    "ella_sampler": ella_sampler,
    "ella_t5_embeds": ella_t5_embeds,
    "diffusers_model_loader": diffusers_model_loader,
    "diffusers_sampler": diffusers_sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ella_model_loader": "ELLA Model Loader",
    "ella_sampler": "ELLA Sampler",
    "ella_t5_embeds": "ELLA T5 Embeds",
    "diffusers_model_loader": "Diffusers Model Loader",
    "diffusers_sampler": "Diffusers Sampler"

}
