from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import tyro
from typing import Optional
from src.config.inference_config import InferenceConfig
from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

app = FastAPI()


class GenerateRequest(BaseModel):
    source: str
    driving: str
    output_dir: str = "output"
    flag_do_torch_compile: bool = False


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


@app.post("/api/v1/generate")
async def process_video(request: GenerateRequest):
    args = tyro.cli(ArgumentConfig)
    args.source = request.source
    args.driving = request.driving
    args.output_dir = request.output_dir
    args.flag_do_torch_compile = request.flag_do_torch_compile

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    result, _ = live_portrait_pipeline.execute(args)

    return {
        "status": "success",
        "output": result
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
