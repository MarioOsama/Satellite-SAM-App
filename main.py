import numpy as np
import ast
import uuid
import fastapi
from fastapi import UploadFile, File
from PIL import Image
import io
from fastsam import FastSAM, FastSAMPrompt
from utils.tools import convert_box_xywh_to_xyxy

app = fastapi.FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Satellite-FastSAM API!"}


@app.post("/predict")
async def predict(img_file: UploadFile = File(...), device: str = "cuda", retina: bool = True, imgsz: int = 1024,
                  conf: float = 0.4, iou: float = 0.9, point_prompt: str = "[[0,0]]", box_prompt: str = "[[0,0,0,0]]",
                  point_label: str = "[0]", text_prompt: str = None):
    """
       Endpoint to predict the segmentation of an uploaded image using the FastSAM model.

       Parameters:
       img_file (UploadFile): The image file to be processed.
       device (str): The device to use for processing (default is "cuda").
       retina (bool): Whether to draw high-resolution segmentation masks (default is True).
       imgsz (int): The size of the image (default is 1024).
       conf (float): The object confidence threshold (default is 0.4).
       iou (float): The IoU threshold for filtering the annotations (default is 0.9).
       point_prompt (str): The point prompt for the FastSAM model (default is "[[0,0]]").
       box_prompt (str): The box prompt for the FastSAM model (default is "[[0,0,0,0]]").
       point_label (str): The point label for the FastSAM model (default is "[0]").
       text_prompt (str): The text prompt for the FastSAM model (default is None).

       Returns:
       list: Processed image.
       """

    # Generate a unique filename
    img_file.filename = generate_filename()

    # Read the contents of the uploaded file
    contents = await read_file_contents(img_file)

    point_prompt, box_prompt, point_label = process_prompts(point_prompt, box_prompt, point_label)

    model = load_model("results/best.pt")

    # Open the uploaded image with PIL
    uploaded_image = open_image(contents)

    # Call the model with the provided arguments
    everything_results = call_model(model, uploaded_image, device, retina, imgsz, conf, iou)

    prompt_process, bboxes, points, point_label, ann = process_results(everything_results, box_prompt, text_prompt,
                                                                       point_prompt, point_label, uploaded_image,
                                                                       device)

    output_path = f'../output/{img_file.filename}'

    # Plot the results and get the image as bytes
    result_image = plot_results(prompt_process, ann, output_path, bboxes, points, point_label)

    # Return the image as a streaming response
    return result_image.tolist()


def generate_filename():
    return f"{uuid.uuid4()}.jpg"


async def read_file_contents(img_file):
    return await img_file.read()


def process_prompts(point_prompt, box_prompt, point_label):
    point_prompt = ast.literal_eval(point_prompt)
    box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
    point_label = ast.literal_eval(point_label)
    return point_prompt, box_prompt, point_label


def load_model(model_path):
    return FastSAM(model_path)


def open_image(contents):
    uploaded_image = Image.open(io.BytesIO(contents))
    return uploaded_image.convert("RGB")


def call_model(model, uploaded_image, device, retina, imgsz, conf, iou):
    return model(uploaded_image, device=device, retina_masks=retina, imgsz=imgsz, conf=conf, iou=iou)


def process_results(everything_results, box_prompt, text_prompt, point_prompt, point_label, uploaded_image, device):
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(uploaded_image, everything_results, device=device)
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
        ann = prompt_process.box_prompt(bboxes=box_prompt)
        bboxes = box_prompt
    elif text_prompt is not None:
        ann = prompt_process.text_prompt(text=text_prompt)
    elif point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
        points = point_prompt
        point_label = point_label
    else:
        ann = prompt_process.everything_prompt()
    return prompt_process, bboxes, points, point_label, ann


def plot_results(prompt_process, ann, output_path, bboxes, points, point_label) -> np.ndarray:
    result = prompt_process.plot(
        annotations=ann,
        output_path=output_path,
        bboxes=bboxes,
        points=points,
        point_label=point_label,
        withContours=False,
        better_quality=False,
    )

    return result
