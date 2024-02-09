import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''
import argparse
import ast
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
#from diffusers import DiffusionPipeline

story="In a bustling coffee shop, Alice, a young woman with blonde hair and a green apron, prepared to brew coffee. She approached the sleek, silver coffee machine, its buttons gleaming under the shop's warm lights. With practiced ease, she selected the finest beans and filled the grinder, the aroma filling the air. As the machine whirred to life, customers eagerly awaited their favorite brews. With each cup poured, the atmosphere buzzed with caffeine-fueled chatter and the aroma of freshly brewed coffee. Alice's skillful hands and the dependable machine transformed the shop into a haven for coffee lovers, one cup at a time."
def ExtractProtagonist(story, file_path):
    answer='''[
    {
        "id": 1,
        "name": "Alice",
        "description": "A young woman with blonde hair, wearing a green apron."
    },
    {
        "id": 2,
        "name": "Coffee Shop",
        "description": "Bustling with warm lights and a sleek, silver coffee machine."
    }
    ]'''
    f = open(file_path, "w")
    f.write(answer)
    f.close()
    protagonists_places_dict = {}
    protagonists_places_dict["Alice"] = "A young woman with blonde hair, wearing a green apron."
    protagonists_places_dict["Coffee Shop"] = "Bustling with warm lights and a sleek, silver coffee machine."
    return protagonists_places_dict

def ExtractAProtagonist(story, file_path):
    answer='''[
    {
        "id": 1,
        "name": "Alice",
        "description": "A young woman with blonde hair, wearing a green apron."
    }
    ]'''
    protagonists_places_dict = {}
    protagonists_places_dict["Alice"] = "A young woman with blonde hair, wearing a green apron."
    f = open(file_path, "w")
    f.write(answer)
    f.close()
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
    answer='''[
    {
        "video segment id": 1, 
        "character/place id": [2]
    },
    {
        "video segment id": 2, 
        "character/place id": [1]
    },
    {
        "video segment id": 3, 
        "character/place id": [0]
    },
    {
        "video segment id": 4, 
        "character/place id": [0]
    },
    {
        "video segment id": 5, 
        "character/place id": [0]
    },
    {
        "video segment id": 6, 
        "character/place id": [0]
    },
    {
        "video segment id": 7,
        "character/place id": [1]
    },
    {
        "video segment id": 8,
        "character/place id": [0]
    },
    {
        "video segment id": 9,
        "character/place id": [0]
    },
    {
        "video segment id": 10,
        "character/place id": [0]
    },
    {
        "video segment id": 11,
        "character/place id": [1]
    },
    {
        "video segment id": 12,
        "character/place id": [0]
    },
    {
        "video segment id": 13,
        "character/place id": [0]
    }
    ]'''
    f = open(file_path, "w")
    f.write(answer)
    f.close()
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
    answer='''[
    {
    "video fragment id": 1,
    "video fragment description": "Busy coffee shop scene."
    },
    {
    "video fragment id": 2,
    "video fragment description": "Close-up: Alice's hands."
    },
    {
    "video fragment id": 3,
    "video fragment description": "Grinder fills with aroma."
    },
    {
    "video fragment id": 4,
    "video fragment description": "Coffee machine whirs."
    },
    {
    "video fragment id": 5,
    "video fragment description": "Steam rises from machine."
    },
    {
    "video fragment id": 6,
    "video fragment description": "Customers wait for orders."
    },
    {
    "video fragment id": 7,
    "video fragment description": "Alice pours coffee."
    },
    {
    "video fragment id": 8,
    "video fragment description": "Aroma intensifies."
    },
     {
    "video fragment id": 9,
    "video fragment description": "Customers take first sips."
    },
    {
    "video fragment id": 10,
    "video fragment description": "Satisfied expressions."
    },
    {
    "video fragment id": 11,
    "video fragment description": "Final shot: Alice smiles."
    },
    {
    "video fragment id": 12,
    "video fragment description": "Customer served."
    },
    {
    "video fragment id": 13,
    "video fragment description": "Coffee machine hums."
    }
    ]'''
    f = open(file_path, "w")
    f.write(answer)
    f.close()
    video_list = ["Busy coffee shop scene.","Close-up: Alice's hands.","Grinder fills with aroma.","Coffee machine whirs.","Steam rises from machine.","Customers wait for orders.",
                  "Alice pours coffee.", "Aroma intensifies.","Customers take first sips.","Satisfied expressions.","Final shot: Alice smiles.","Customer served.","Coffee machine hums."]
    return video_list

def translate_video_script(video_list, file_path):
            answer = '''[
            {
                "序号": 1,
                "描述": "繁忙的咖啡店场景。"
            },
            {
                "序号": 2,
                "描述": "特写：爱丽丝的手。"
            },
            {
                "序号": 3,
                "描述": "磨豆机充满香气。"
            },
            {
                "序号": 4,
                "描述": "咖啡机嗡嗡作响。"
            },
            {
                "序号": 5,
                "描述": "蒸汽从机器中升起。"
            },
            {
                "序号": 6,
                "描述": "顾客等待订单。"
            },
            {
                "序号": 7,
                "描述": "爱丽丝倒咖啡。"
            },
            {
                "序号": 8,
                "描述": "香气加强。"
            },
            {
                "序号": 9,
                "描述": "顾客品尝第一口。"
            },
            {
                "序号": 10,
                "描述": "满意的表情。"
            },
            {
                "序号": 11,
                "描述": "最后镜头：爱丽丝微笑。"
            },
            {
                "序号": 12,
                "描述": "为顾客服务。"
            },
            {
                "序号": 13,
                "描述": "咖啡机嗡嗡作响。"
            }
            ]'''
            f = open(file_path, "w")
            f.write(answer)
            f.close()
            zh_video_list = ["繁忙的咖啡店场景。","特写：爱丽丝的手。","磨豆机充满香气。","咖啡机嗡嗡作响。","蒸汽从机器中升起。","顾客等待订单。","爱丽丝倒咖啡。","香气加强。","顾客品尝第一口。","满意的表情。","最后镜头：爱丽丝微笑。","为顾客服务。","咖啡机嗡嗡作响。"]
            #assert len(video_list) == len(zh_video_list)
            return zh_video_list
    
def time_scripts(video_list, file_path):
            new_video_list = []
            num = 1
            for video in video_list:
                prompt = str(num) + ". " + video
                new_video_list.append(prompt)
                num += 1
            answer='''[
            {
                "video fragment id": 1,
                "time": 3
            },
            {
                "video fragment id": 2,
                "time": 3
            },
            {
                "video fragment id": 3,
                "time": 3
            },
            {
                "video fragment id": 4,
                "time": 1
            },
            {
                "video fragment id": 5,
                "time": 1
            },
            {
                "video fragment id": 6,
                "time": 1
            },
            {
                "video fragment id": 7,
                "time": 1
            },
            {
                "video fragment id": 8,
                "time": 1
            },
            {
                "video fragment id": 9,
                "time": 1
            },
            {
                "video fragment id": 10,
                "time": 1
            },
            {
                "video fragment id": 11,
                "time": 2
            },
            {
                "video fragment id": 12,
                "time": 1
            },
            {
                "video fragment id": 13,
                "time": 1
            }
            ]'''
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




def main(args):
    story_path = args.story_path
    save_script_path = os.path.join(story_path.rsplit('/', 1)[0], "script")
    if not os.path.exists(save_script_path):
            os.makedirs(save_script_path)
    with open(story_path, "r") as story_file:
        story = story_file.read()
        
    # summerize protagonists and places
    protagonists_places_file_path = os.path.join(save_script_path, "protagonists_places.txt")
    if args.only_one_protagonist:
        character_places = ExtractAProtagonist(story, protagonists_places_file_path)
    else:
        character_places = ExtractProtagonist(story, protagonists_places_file_path)
    print("Protagonists and places OK", flush=True)
    
    # make script
    script_file_path = os.path.join(save_script_path, "video_prompts.txt")
    video_list = split_story(story, script_file_path)
    video_list = patch_story_scripts(story, video_list, script_file_path)
    video_list = refine_story_scripts(video_list, script_file_path)
    print("Scripts OK", flush=True)
    
    # think about the protagonist in each scene
    reference_file_path = os.path.join(save_script_path, "protagonist_place_reference.txt")
    reference_lists = protagonist_place_reference1(video_list, character_places, reference_file_path)
    print("Reference protagonist OK", flush=True)
    
    # translate the English script to Chinese
    zh_file_path = os.path.join(save_script_path, "zh_video_prompts.txt")
    zh_video_list = translate_video_script(video_list, zh_file_path)
    print("Translation OK", flush=True)
    
    # schedule the time of script
    time_file_path = os.path.join(save_script_path, "time_scripts.txt")
    time_list = time_scripts(video_list, time_file_path)
    print("Time script OK", flush=True)
    
    # make reference image
    #base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16,use_safetensors=True, variant="fp16").to("cuda")
    #refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=base.text_encoder_2,vae=base.vae,torch_dtype=torch.float16,use_safetensors=True,variant="fp16",).to("cuda")
    model_id = "runwayml/stable-diffusion-v1-5"
    base = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    ref_dir_path = os.path.join(story_path.rsplit('/', 1)[0], "ref_img")
    if not os.path.exists(ref_dir_path):
            os.makedirs(ref_dir_path)
    for key, value in character_places.items():
        prompt = key + ", " + value
        img_path = os.path.join(ref_dir_path, key + ".jpg")
        image = base(prompt=prompt).images[0]
        #image = refiner(prompt=prompt, image=image[None, :]).images[0]
    print("Reference image OK",img_path, flush=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Vlogger/configs/vlog_write_script.yaml")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    main(omega_conf)
