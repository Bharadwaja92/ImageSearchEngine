from fastapi import FastAPI, Query
from app.ImageSearchEngine import ImageSearchEngine
from fastapi.responses import PlainTextResponse

app = FastAPI()
config_file = 'config/creds.yaml'
search_engine = ImageSearchEngine(config_file)

@app.get("/search")
def search_images(query: str = Query(..., description="User input for image search")):
    results = search_engine.retrieve_images(query)
    payloads = [r.payload for r in results]
    ids = [str(r.id).zfill(4) for r in results]
    return {"results": [{"id": i, "payload": p} for i, p in zip(ids, payloads)]}

@app.get("/explanations")
def explain_images(query: str = Query(..., description="User input")):
    # return PlainTextResponse(content='Not able to show explanations because of Size constraints.', media_type="text/plain")
    results = search_engine.retrieve_images(query)
    success, response = search_engine.get_explanations(query, results)
    if success:
        return response.model_dump()
    return PlainTextResponse(content=response, media_type="text/plain")