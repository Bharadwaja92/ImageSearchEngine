import pytest
from pathlib import Path
import sys
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from app.ImageSearchEngine import ImageSearchEngine

@pytest.fixture(scope="module")
def image_search_engine():
    config_path = Path(__file__).resolve().parents[1] / "config" / "creds.yaml"
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    return ImageSearchEngine(config_file_path=config_path)

def test_retrieve_images(image_search_engine):
    result = image_search_engine.retrieve_images("pets")
    assert len(result) > 0
    for item in result:
        assert hasattr(item, "id")
        assert hasattr(item, "payload")

def test_get_explanations(image_search_engine):
    query = "pets"
    results = image_search_engine.retrieve_images(query)
    success, explanations = image_search_engine.get_explanations(query, results)
    assert success is True
    assert hasattr(explanations, "image1")
