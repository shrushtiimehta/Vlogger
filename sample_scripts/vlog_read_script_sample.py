import torch
import ast
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import sys
try:
    import utils
    from diffusion import create_diffusion
except:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion
import argparse
import torchvision
from PIL import Image
from einops import rearrange
from models import get_models
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils import mask_generation_before
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from vlogger.videofusion import fusion
from vlogger.videocaption import captioning
from vlogger.videoaudio import make_audio, merge_video_audio, concatenate_videos
from vlogger.STEB.model_transform import ip_scale_set, ip_transform_model, tca_transform_model

#story="In a bustling coffee shop, Alice, a young woman with blonde hair and a green apron, prepared to brew coffee. She approached the sleek, silver coffee machine, its buttons gleaming under the shop's warm lights. With practiced ease, she selected the finest beans and filled the grinder, the aroma filling the air. As the machine whirred to life, customers eagerly awaited their favorite brews. With each cup poured, the atmosphere buzzed with caffeine-fueled chatter and the aroma of freshly brewed coffee. Alice's skillful hands and the dependable machine transformed the shop into a haven for coffee lovers, one cup at a time."

def ExtractProtagonist(story, file_path):
    protagonists_places_dict = {}
    protagonists_places_dict["Alice"] = "A young woman with blonde hair, wearing a green apron."
    protagonists_places_dict["Coffee Shop"] = "Bustling with warm lights and a sleek, silver coffee machine."
    return protagonists_places_dict

def ExtractAProtagonist(story, file_path):
    protagonists_places_dict = {}
    protagonists_places_dict["Alice"] = "A young woman with blonde hair, wearing a green apron."
    return protagonists_places_dict

def protagonist_place_reference1(video_list, character_places, file_path):
    new_video_list = []
    num = 1
    for video in video_list:
        prompt = str(num) + ". " + video
        new_video_list.append(prompt)
        num += 1
    key_list = []
    i = 1
    for key, value in character_places.items():
        key_list.append(str(i) + ". " + key)
        i += 1
    reference_list = []
    protagonists_places_reference = [{"video segment id": 1, "character/place id": [2]},{"video segment id": 2, "character/place id": [1]},{"video segment id": 3, "character/place id": [0]},{"video segment id": 4, "character/place id": [0]},{"video segment id": 5, "character/place id": [0]},
                                 {"video segment id": 6, "character/place id": [0]},{"video segment id": 7,"character/place id": [1]},{"video segment id": 8,"character/place id": [0]},{"video segment id": 9,"character/place id": [0]},{"video segment id": 10,"character/place id": [0]},{"video segment id": 11,"character/place id": [1]},{"video segment id": 12,"character/place id": [0]},{"video segment id": 13,"character/place id": [0]}]
    for key, value in character_places.items():
        if key.lower() in prompt:
          protagonists_places_reference[i]["character/place id"] = [1]
    for protagonist_place_reference in protagonists_places_reference:
        reference_list.append(protagonist_place_reference["character/place id"])
    return reference_list

def split_story(story, file_path):
    video_list = ["Opening shot of a bustling coffee shop with warm lights and customers chatting. Alice, a young woman with blonde hair and a green apron, stands behind the sleek, silver coffee machine, preparing to brew coffee.", "Close-up of Alice's hands as she selects the finest coffee beans and fills the grinder. The aroma of coffee fills the air, blending with the background chatter of customers.",
                  "Wide shot of the coffee machine as it whirs to life, steam rising from the spout. Customers eagerly await their orders, glancing at the menu board and chatting with baristas.","Over-the-shoulder shot of Alice as she expertly pours freshly brewed coffee into cups, her hands moving with precision. The aroma intensifies, enveloping the shop in a comforting scent.",
                  "Cut to a montage of satisfied customers taking their first sips of coffee, their faces lighting up with delight. The atmosphere buzzes with energy and contentment.",  "Final shot of the coffee shop, with Alice smiling behind the counter as she serves another customer. The dependable coffee machine hums softly in the background, symbolizing the shop's role as a haven for coffee lovers."]
    return video_list


def patch_story_scripts(story, video_list, file_path):
    video_list = ["Bustling coffee shop with Alice behind sleek coffee machine.","Close-up of Alice's hands filling grinder, aroma fills air.","Coffee machine whirs, steam rises, customers await orders.",
    "Alice expertly pours coffee into cups, aroma intensifies.","Montage of satisfied customers taking first sips of coffee.","Final shot: Alice smiles, serves customer, coffee machine hums."]
    return video_list


def refine_story_scripts(video_list, file_path):
    video_list = ["Busy coffee shop scene.","Close-up: Alice's hands.","Grinder fills with aroma.","Coffee machine whirs.","Steam rises from machine.","Customers wait for orders.",
                  "Alice pours coffee.", "Aroma intensifies.","Customers take first sips.","Satisfied expressions.","Final shot: Alice smiles.","Customer served.","Coffee machine hums."]
    return video_list


def time_scripts(video_list, file_path):
            new_video_list = []
            num = 1
            for video in video_list:
                prompt = str(num) + ". " + video
                new_video_list.append(prompt)
                num += 1
            time_scripts = [{"video fragment id": 1,"time": 3},{"video fragment id": 2,"time": 3},{"video fragment id": 3,"time": 3},{"video fragment id": 4,"time": 1},{"video fragment id": 5,"time": 1},{"video fragment id": 6,"time": 1},
             {"video fragment id": 7,"time": 1},{"video fragment id": 8,"time": 1},{"video fragment id": 9,"time": 1},{"video fragment id": 10,"time": 1},{"video fragment id": 11,"time": 2},{"video fragment id": 12,"time": 1},{"video fragment id": 13,"time": 1}]
            time_list = []
            for time_script in time_scripts:
                time = time_script["time"]
                if time > 10:
                    time = 10
                time_list.append(time)
            assert len(time_list) == len(video_list)
            return time_list

def readscript(script_file_path):
    with open(script_file_path, "r", encoding='utf-8') as f:
        script = f.read()
        video_fragments = ast.literal_eval(script)
        video_list = []
        for video_fragment in video_fragments:
            video_list.append(video_fragment["video fragment description"])
    return video_list


def readzhscript(zh_file_path):
    with open(zh_file_path, "r", encoding='utf-8') as f: 
        script = f.read()
        video_fragments = ast.literal_eval(script)
        video_list = []
        for video_fragment in video_fragments:
            video_list.append(video_fragment["描述"])
    return video_list


def readtimescript(time_file_path):
    with open(time_file_path, "r", encoding='utf-8') as f:
        time_scripts = f.read()
        time_scripts = ast.literal_eval(time_scripts)
        time_list = []
        for time_script in time_scripts:
            frames = time_script["time"]
            time_list.append(frames)
    return time_list


def readprotagonistscript(protagonist_file_path):
    with open(protagonist_file_path, "r", encoding='utf-8') as f:
        protagonist_scripts = f.read()
        protagonist_scripts = ast.literal_eval(protagonist_scripts)
        protagonists_places_dict = {}
        for protagonist_script in protagonist_scripts:
            protagonists_places_dict[protagonist_script["name"]] = protagonist_script["description"]
    return protagonists_places_dict


def readreferencescript(video_list, character_places, reference_file_path):
    new_video_list = []
    num = 1
    for video in video_list:
        prompt = str(num) + ". " + video
        new_video_list.append(prompt)
        num += 1
    key_list = []
    i = 1
    for key, value in character_places.items():
        key_list.append(str(i) + ". " + key)
    with open(reference_file_path, "r", encoding='utf-8') as f:
        reference_file = f.read()
        reference_list = []
        protagonists_places_reference = ast.literal_eval(reference_file)
        for i, prompt in enumerate(video_list):
            prompt = prompt.lower()
            for j, key in enumerate(key_list):
                if key.lower() in prompt:
                    protagonists_places_reference[i]["character/place id"] = [j + 1]

        for protagonist_place_reference in protagonists_places_reference:
            reference_list.append(protagonist_place_reference["character/place id"])
    return reference_list




def auto_inpainting(args, 
                    video_input, 
                    masked_video, 
                    mask, 
                    prompt, 
                    image, 
                    vae, 
                    text_encoder, 
                    image_encoder, 
                    diffusion, 
                    model, 
                    device,
                    ):
    image_prompt_embeds = None
    if prompt is None:
        prompt = ""
    if image is not None:
        clip_image = CLIPImageProcessor()(images=image, return_tensors="pt").pixel_values
        clip_image_embeds = image_encoder(clip_image.to(device)).image_embeds
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds).to(device)
        image_prompt_embeds = torch.cat([clip_image_embeds, uncond_clip_image_embeds], dim=0)
        image_prompt_embeds = rearrange(image_prompt_embeds, '(b n) c -> b n c', b=2).contiguous()
        model = ip_scale_set(model, args.ref_cfg_scale)
        if args.use_fp16:
            image_prompt_embeds = image_prompt_embeds.to(dtype=torch.float16)
    b, f, c, h, w = video_input.shape
    latent_h = video_input.shape[-2] // 8
    latent_w = video_input.shape[-1] // 8

    if args.use_fp16:
        z = torch.randn(1, 4, 16, latent_h, latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, 16, latent_h, latent_w, device=device) # b,c,f,h,w

    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
    masked_video = torch.cat([masked_video] * 2)
    mask = torch.cat([mask] * 2)
    z = torch.cat([z] * 2)
    prompt_all = [prompt] + [args.negative_prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                        class_labels=None, 
                        cfg_scale=args.cfg_scale,
                        use_fp16=args.use_fp16,
                        ip_hidden_states=image_prompt_embeds)
    
    # Sample images:
    samples = diffusion.ddim_sample_loop(model.forward_with_cfg, 
                                         z.shape, 
                                         z, 
                                         clip_denoised=False, 
                                         model_kwargs=model_kwargs, 
                                         progress=True, 
                                         device=device,
                                         mask=mask, 
                                         x_start=masked_video, 
                                         use_concat=True,
                                         )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip


def main(args):
    # Setup PyTorch:
    if args.seed:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    model = get_models(args).to(device)
    model = tca_transform_model(model).to(device)
    model = ip_transform_model(model).to(device)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    if args.use_compile:
        model = torch.compile(model)

    ckpt_path = args.ckpt 
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['ema']
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    text_encoder = text_encoder = TextEmbedder(args.pretrained_model_path).to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(device)
    if args.use_fp16:
        print('Warnning: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
    print("model ready!\n", flush=True)
    
    
    # load protagonist script
    character_places = readprotagonistscript(args.protagonist_file_path)
    print("protagonists ready!", flush=True)

    # load script
    video_list = readscript(args.script_file_path)
    print("video script ready!", flush=True)
    
    # load reference script
    reference_lists = readreferencescript(video_list, character_places, args.reference_file_path)
    print("reference script ready!", flush=True)
    
    # load zh script
    zh_video_list = readzhscript(args.zh_script_file_path)
    print("zh script ready!", flush=True)
    
    # load time script
    key_list = []
    for key, value in character_places.items():
        key_list.append(key)
    time_list = readtimescript(args.time_file_path)
    print("time script ready!", flush=True)
    

    # generation begin
    sample_list = []
    for i, text_prompt in enumerate(video_list):
        sample_list.append([])
        for time in range(time_list[i]):
            if time == 0:
                print('Generating the ({}) prompt'.format(text_prompt), flush=True)
                if reference_lists[i][0] == 0 or reference_lists[i][0] > len(key_list):
                    pil_image = None
                else:
                    pil_image = Image.open(args.reference_image_path[reference_lists[i][0] - 1])
                    pil_image.resize((256, 256))
                video_input = torch.zeros([1, 16, 3, args.image_size[0], args.image_size[1]]).to(device)
                mask = mask_generation_before("first0", video_input.shape, video_input.dtype, device) # b,f,c,h,w
                masked_video = video_input * (mask == 0)
                samples = auto_inpainting(args, 
                                          video_input, 
                                          masked_video, 
                                          mask, 
                                          text_prompt, 
                                          pil_image, 
                                          vae, 
                                          text_encoder, 
                                          image_encoder, 
                                          diffusion, 
                                          model, 
                                          device,
                                          )
                sample_list[i].append(samples)
            else:
                if sum(video.shape[0] for video in sample_list[i]) / args.fps >= time_list[i]:
                    break
                print('Generating the ({}) prompt'.format(text_prompt), flush=True)
                if reference_lists[i][0] == 0 or reference_lists[i][0] > len(key_list):
                    pil_image = None
                else:
                    pil_image = Image.open(args.reference_image_path[reference_lists[i][0] - 1])
                    pil_image.resize((256, 256))
                pre_video = sample_list[i][-1][-args.researve_frame:]
                f, c, h, w = pre_video.shape
                lat_video = torch.zeros(args.num_frames - args.researve_frame, c, h, w).to(device)
                video_input = torch.concat([pre_video, lat_video], dim=0)
                video_input = video_input.to(device).unsqueeze(0)
                mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device)
                masked_video = video_input * (mask == 0)
                video_clip = auto_inpainting(args, 
                                             video_input, 
                                             masked_video, 
                                             mask, 
                                             text_prompt, 
                                             pil_image, 
                                             vae, 
                                             text_encoder, 
                                             image_encoder, 
                                             diffusion, 
                                             model, 
                                             device,
                                             )
                sample_list[i].append(video_clip[args.researve_frame:])
                print(video_clip[args.researve_frame:].shape)

        # transition
        if args.video_transition and i != 0:
            video_1 = sample_list[i - 1][-1][-1:]
            video_2 = sample_list[i][0][:1]
            f, c, h, w = video_1.shape
            video_middle = torch.zeros(args.num_frames - 2, c, h, w).to(device)
            video_input = torch.concat([video_1, video_middle, video_2], dim=0)
            video_input = video_input.to(device).unsqueeze(0)
            mask = mask_generation_before("onelast1", video_input.shape, video_input.dtype, device)
            masked_video = masked_video = video_input * (mask == 0)
            video_clip = auto_inpainting(args, 
                                         video_input, 
                                         masked_video, 
                                         mask, 
                                         "smooth transition, slow motion, slow changing.", 
                                         pil_image, 
                                         vae, 
                                         text_encoder, 
                                         image_encoder, 
                                         diffusion, 
                                         model, 
                                         device,
                                         )
            sample_list[i].insert(0, video_clip[1:-1])

        # save videos
        samples = torch.concat(sample_list[i], dim=0)
        samples = samples[0: time_list[i] * args.fps]
        if not os.path.exists(args.save_origin_video_path):
            os.makedirs(args.save_origin_video_path)
        video_ = ((samples * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        torchvision.io.write_video(args.save_origin_video_path + "/" + f"{i}" + '.mp4', video_, fps=args.fps)
    
    # post processing
    fusion(args.save_origin_video_path)
    captioning(args.script_file_path, args.zh_script_file_path, args.save_origin_video_path, args.save_caption_video_path)
    fusion(args.save_caption_video_path)
    make_audio(args.script_file_path, args.save_audio_path)
    merge_video_audio(args.save_caption_video_path, args.save_audio_path, args.save_audio_caption_video_path)
    concatenate_videos(args.save_audio_caption_video_path)
    print('final video save path {}'.format(args.save_audio_caption_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Vlogger/configs/vlog_read_script_sample.yaml")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    save_path = omega_conf.save_path
    save_origin_video_path = os.path.join(save_path, "origin_video")
    save_caption_video_path = os.path.join(save_path.rsplit('/', 1)[0], "caption_video")
    save_audio_path = os.path.join(save_path.rsplit('/', 1)[0], "audio")
    save_audio_caption_video_path = os.path.join(save_path.rsplit('/', 1)[0], "audio_caption_video")
    if omega_conf.sample_num is not None:
        for i in range(omega_conf.sample_num):
            omega_conf.save_origin_video_path = save_origin_video_path + f'-{i}'
            omega_conf.save_caption_video_path = save_caption_video_path + f'-{i}'
            omega_conf.save_audio_path = save_audio_path + f'-{i}'
            omega_conf.save_audio_caption_video_path = save_audio_caption_video_path + f'-{i}'
            omega_conf.seed += i
            main(omega_conf)
    else:
        omega_conf.save_origin_video_path = save_origin_video_path
        omega_conf.save_caption_video_path = save_caption_video_path
        omega_conf.save_audio_path = save_audio_path
        omega_conf.save_audio_caption_video_path = save_audio_caption_video_path
        main(omega_conf)
