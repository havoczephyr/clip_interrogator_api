from fastapi import FastAPI, File, UploadFile
import os
import shutil
from PIL import Image

from clip_interrogator import Interrogator, Config

# image_path = "living_room.jpeg"




app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        try:
            with open(f"demo_images/{file.filename}", 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
                image = Image.open(f"demo_images/{file.filename}").convert('RGB')
                ci = Interrogator(Config(clip_model_name="ViT-B/16", flavor_intermediate_count=256))
        finally:
            os.remove(f"demo_images/{file.filename}")
            return {"message": ci.interrogate(image)}
