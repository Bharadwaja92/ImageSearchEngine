import logging
import json
import torch
import requests
from typing import List, Tuple, Union, Any
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for LLM output
class ImageExplanation(BaseModel):
    Caption: str
    Explanation: str

class ExplanationResponse(BaseModel):
    image1: ImageExplanation
    image2: ImageExplanation
    image3: ImageExplanation
    image4: ImageExplanation
    image5: ImageExplanation

class ImageSearchEngine:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        self.QDRANT_URL = config['QDRANT_URL']
        self.LLM_URL = config['LLM_URL']
        self.QDRANT_COLLECTION_NAME = config['QDRANT_COLLECTION_NAME']
        self.PROMPT_TEMPLATE_PATH = config['PROMPT_TEMPLATE_PATH']
        
        self.client = QdrantClient(url=self.QDRANT_URL)
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        with open(self.PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        
        logger.info("ImageSearchEngine initialized.")

    def retrieve_images(self, user_input: str) -> List[Any]:
        """
        Converts the input text into an embedding using CLIP and retrieves similar images from Qdrant.

        Args:
        user_input (str): The natural language search query.

        Returns:
        List[Any]: Top 5 Qdrant points (images) with payloads.
        """
        text_inputs = self.clip_processor(text=user_input, return_tensors='pt', padding=True)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.squeeze().tolist()

        try:
            results = self.client.query_points(
                collection_name=self.QDRANT_COLLECTION_NAME,
                query=text_embeddings,
                with_payload=True,
                limit=5
            ).points
            logger.info(f"Connected to Qdrant successfully.")
            logger.info(f"Retrieved {len(results)} images.")
        except Exception as e:
            logger.info(f"Got error {e} while retrieving.")
            raise ValueError(f"Got error {e} while retrieving.")
        
        return results

    def get_explanations(self, user_query: str, search_results: List[Any]) -> Tuple[bool, Union[ExplanationResponse, str]]:
        """
        Generate LLM explanations for retrieved image payloads.

        Args:
            user_query (str): The original search query from the user.
            search_results (List[Any]): List of search results from Qdrant.

        Returns:
            Tuple[bool, Union[ExplanationResponse, str]]: 
                - (True, ExplanationResponse) on successful parsing
                - (False, raw_output) on failure
        """
        image_data = [s.payload for s in search_results]
        image_sections = [{
            'caption': img.get('caption', ''), 
            'scene': img.get('scenes', ''), 
            'objects': img.get('objects', [])
        } for img in image_data]

        prompt = self.prompt_template.format(user_query=user_query, image_data=image_sections)
        logger.info(f'prompt = {prompt}')

        try:
            response = requests.post(
                self.LLM_URL,
                json={"model": "llama3", "prompt": prompt, "stream": False}
            )
            logger.info(f'response = {response}')
            raw_output = response.json()["response"].strip()
            logger.info(f'raw_output = {raw_output}')
            json_start = raw_output.index("{")
            json_obj = json.loads(raw_output[json_start:])
            parsed = ExplanationResponse(**json_obj)
            return True, parsed
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error("Failed to parse LLM output")
            return False, raw_output
