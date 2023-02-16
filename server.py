from pathlib import Path
from pydantic import BaseModel

import os
import cv2
from numpy import ndarray
from base64 import b64encode

import asyncio
from io import BytesIO
from datetime import datetime

from functools import wraps
from typing import NamedTuple, Callable, Coroutine, Tuple, Dict, Any

from fastapi.responses import Response
from fastapi import FastAPI, HTTPException, Depends

application = FastAPI()

BASE_DIR = Path(__file__).parent
IMAGES_PATH = BASE_DIR / 'images'

if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)


class ImageDoesNotExistException(Exception):
    pass


class ImageCannotBeSavedException(Exception):
    pass


def sync_to_async(function: Callable) -> Coroutine:

    @wraps(function)
    async def wrapper(*args: Tuple, **kwargs: Dict) -> Any:
        return await asyncio.to_thread(function, *args, **kwargs)

    return wrapper


class GenerateImageInput(BaseModel):
    image_name: str
    client_name: str
    pix: str


def get_image_path(image_name: str) -> str:
    image_name = image_name.lower()

    for image in os.listdir(IMAGES_PATH):
        if image.lower().startswith(image_name):
            return str(IMAGES_PATH / image)

    raise ImageDoesNotExistException(f"{image_name} does not exist")


class DataToPutOnImage(NamedTuple):
    text: str
    position: Tuple
    size: float


def put_label_on_image(
    image: ndarray,
    data_to_put_on_image: DataToPutOnImage
) -> None:

    cv2.putText(
        image,
        data_to_put_on_image.text,
        data_to_put_on_image.position,
        cv2.FONT_HERSHEY_DUPLEX,
        data_to_put_on_image.size,
        (0, 0, 0),
        2
    )


def put_labels_on_image(
    image: ndarray,
    image_data: GenerateImageInput
) -> None:

    put_label_on_image(image, DataToPutOnImage(
        text=f"Cliente: {image_data.client_name}".upper(),
        position=(10, 410),
        size=0.8
    ))

    put_label_on_image(image, DataToPutOnImage(
        text=image_data.pix,
        position=(60, 500),
        size=3
    ))

    put_label_on_image(image, DataToPutOnImage(
        text='R$',
        position=(10, 500),
        size=1
    ))

    put_label_on_image(image, DataToPutOnImage(
        text=datetime.now().strftime("Pago em %H:%M:%S %d/%m/%Y"),
        position=(10, 540),
        size=0.75
    ))


def write_image_to_buffer(image: ndarray) -> BytesIO:
    is_success, encoded_image = cv2.imencode('.jpg', image)

    if is_success:
        return BytesIO(encoded_image)

    raise ImageCannotBeSavedException()


def convert_buffer_to_base64(buffer: BytesIO) -> str:
    return b64encode(buffer.getvalue()).decode('utf-8')


@sync_to_async
def generate_image_in_base64(image_data: GenerateImageInput) -> BytesIO:
    image = cv2.imread(get_image_path(image_data.image_name))
    put_labels_on_image(image, image_data)
    return convert_buffer_to_base64(write_image_to_buffer(image))


@application.get('/generate_image/')
async def generate_image_controller(
    image_data: GenerateImageInput = Depends()
) -> Response:

    try:
        base64_image = await generate_image_in_base64(image_data)
    except (ImageDoesNotExistException, ImageCannotBeSavedException) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=409,
            detail=f"Unexpected error: {str(error)}"
        )
    else:
        return Response(content=base64_image)
