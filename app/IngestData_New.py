import os
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForCausalLM 
import torch
import nltk

from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from deepface import DeepFace

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

coca_model = AutoModelForCausalLM.from_pretrained('microsoft/git-base-coco')
coca_processor = AutoProcessor.from_pretrained('microsoft/git-base-coco')

client = QdrantClient(url='http://localhost:6333')

# client.create_collection(
#     collection_name='images',
#     vectors_config=VectorParams(size=512, distance=Distance.COSINE)
# )

def get_image_embedding(image):
    image = image.convert('RGB')
    inputs = clip_processor(images=image, return_tensors='pt')
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)

    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    image_embedding = image_embedding.squeeze().tolist()
    return image_embedding

def get_image_caption(image):
    image = image.convert('RGB')
    inputs = coca_processor(images=image, return_tensors='pt')
    caption_encoded = coca_model.generate(**inputs, max_length=30, num_beams=4)
    caption = coca_processor.batch_decode(caption_encoded, skip_special_tokens=True)[0]
    return caption

def get_caption_embedding(caption: str):
    text_inputs = clip_processor(text=caption, return_tensors='pt', padding=True)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_inputs)

    caption_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    caption_embeddings = caption_embeddings.squeeze().tolist()

    return caption_embeddings

def get_key_words(caption: str):
    caption_tokens = nltk.word_tokenize(caption)
    pos_tagged = nltk.pos_tag(caption_tokens)
    keywords = [word for word, tag in pos_tagged if (tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'))]
    return keywords

scene_model = resnet18(num_classes=365)
scene_model.load_state_dict(torch.hub.load_state_dict_from_url(
    'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
)['state_dict'], strict=False)
scene_model.eval()

with open('categories_places365.txt') as f: 
    scene_classes = [line.strip().split(' ')[0][3:] for line in f]

scene_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_scene(img: Image.Image):
    img_t = scene_transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = scene_model(img_t)
    topk = torch.topk(logits, k=5, dim=1)
    top_indices = topk.indices[0].tolist()
    # print(top_indices)
    top_scenes = [scene_classes[t] for t in top_indices]
    return top_scenes

yolo_model = YOLO('yolov8n.pt')

def get_objects(image_loc: str):
    results = yolo_model(source=image_loc, verbose=False)[0]
    return list(set([yolo_model.model.names[int(cls)] for cls in results.boxes.cls]))

def get_emotions(image_loc):
    try:
        emotion = DeepFace.analyze(img_path=image_loc, actions=['emotion'], enforce_detection=False)  # ('emotion', 'age', 'gender', 'race')
        return emotion[0]['dominant_emotion']
    except:
        return 'unknown'

image_names = [f'./images/{img}' for img in sorted(os.listdir('./images/'))]

points = []
for i, image_loc in enumerate(image_names):
    img = Image.open(image_loc)
    # display(img)
    image_embedding = get_image_embedding(img)
    caption = get_image_caption(img)
    print(i+1, caption) 
    image_scenes = get_scene(img)
    objects = get_objects(image_loc=image_loc)

    curr_point = PointStruct(id=i+1,   # CHANGE HERE
                             vector=image_embedding, 
                             payload={
                                 "caption": caption,
                                 "scenes": image_scenes,
                                 "objects": objects
                                 }
                            )
    points.append(curr_point)
    if len(points) == 20:
        print('Ingesting records...')
        client.upsert(
            collection_name='images',
            points=points
        )
        points = []
if points:
    client.upsert(collection_name='images', points=points)



