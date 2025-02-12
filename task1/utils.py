"""
Utilities.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

import datasets
from PIL import Image
from pydantic import BaseModel, field_validator, model_validator
from transformers import AutoProcessor

WORD_LABELS = [
    "OTHER",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]

# Split long sentences in two to avoid token overflow
MAX_WORDS = 225


class Annotation(BaseModel):
    """
    Annotation of single id.
    """

    linking: list[list[int]]
    label: str
    text: str
    word_indices: list[int]

    @field_validator("linking", mode="before")
    def check_bboxes(cls: Any, value: list[list[int]]) -> list[list[int]]:
        """
        Check linking shape.
        """
        if all(len(item) == 2 for item in value):
            return value
        raise ValueError("Invalid linking format")


class Datum(BaseModel):
    """
    Data of a document.
    """

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    basename: str
    image: Image.Image
    annotations: dict[int, Annotation]
    words: list[str]
    word_labels: list[int]
    bboxes: list[list[int]]

    @field_validator("image", mode="before")
    def check_image(cls: Any, value: Image.Image) -> Image.Image:
        """
        Check Image type as pydantic doesn't support natively.
        """
        if isinstance(value, Image.Image):
            return value
        raise ValueError("Invalid image format")

    @field_validator("bboxes", mode="before")
    def check_bboxes(cls: Any, value: list[list[int]]) -> list[list[int]]:
        """
        Check bboxes shape.
        """
        if all(len(item) == 4 for item in value):
            return value
        raise ValueError("Invalid bboxes format")

    @model_validator(mode="after")
    def check_shape(cls, values):  # type: ignore[no-untyped-def]
        """
        Check that words, word_labels and bboxes have the same length.
        """
        if len(values.words) == len(values.word_labels) == len(values.bboxes):
            return values
        raise ValueError("Incorrect shapes")


class SampleMapping(BaseModel):
    """
    Dataclass to map back split word tokens and logits.

    E.g. when there are too many tokens, the tokens are split in
    two overlapping regions.

    Args:
        sample_index: the sample index
        word_start_index: the index of the first word
        word_end_index: the index of the last word
    """

    sample_index: int
    word_start_index: int
    word_end_index: int


Dataset: TypeAlias = dict[str, list[Datum]]


def prepare_data(data: list[Datum], processor: AutoProcessor) -> datasets.Dataset:
    """
    Convert the data into a dataset compatible with the trainer.

    Some of the examples exceeds the token limit of the model.
    The proper way of handling this would be a randomly sampled sliding window during
    training, but instead I opted for splitting the longer examples into two, with
    some overlap in the middle.

    Args:
        data: the dataset
        processor: the processor for the model

    Returns:
        the transformed dataset
    """
    sample_mapping: list[SampleMapping] = []
    encodings: list[dict[str, Any]] = []
    for i, element in enumerate(data):
        n_words = len(element.words)
        if n_words <= MAX_WORDS:
            encodings_ = processor(
                images=element.image,
                text=element.words,
                boxes=element.bboxes,
                word_labels=element.word_labels,
                return_tensors="pt",
                truncation=False,
                padding="max_length",
            )
            assert len(encodings_["labels"][0]) == 512
            encodings.append(encodings_)
            sample_mapping.append(
                SampleMapping(
                    sample_index=i,
                    word_start_index=0,
                    word_end_index=len(element.words),
                )
            )
        else:
            assert n_words <= 2 * MAX_WORDS
            encodings_ = processor(
                images=element.image,
                text=element.words[:MAX_WORDS],
                boxes=element.bboxes[:MAX_WORDS],
                word_labels=element.word_labels[:MAX_WORDS],
                return_tensors="pt",
                truncation=False,
                padding="max_length",
            )
            assert len(encodings_["labels"][0]) == 512
            encodings.append(encodings_)
            sample_mapping.append(
                SampleMapping(
                    sample_index=i,
                    word_start_index=0,
                    word_end_index=MAX_WORDS,
                )
            )
            encodings_ = processor(
                images=element.image,
                text=element.words[-MAX_WORDS:],
                boxes=element.bboxes[-MAX_WORDS:],
                word_labels=element.word_labels[-MAX_WORDS:],
                return_tensors="pt",
                truncation=False,
                padding="max_length",
            )
            assert len(encodings_["labels"][0]) == 512
            encodings.append(encodings_)
            sample_mapping.append(
                SampleMapping(
                    sample_index=i,
                    word_start_index=len(element.words) - MAX_WORDS,
                    word_end_index=len(element.words),
                )
            )

    # Remove batch index as the data collator adds this
    dataset = datasets.Dataset.from_list(
        [{key: value[0] for key, value in entry.items()} for entry in encodings]
    )
    return dataset, sample_mapping


def parse_data(
    dataset_location: str, bounding_box_mode: str = "word", normalize: bool = True
) -> Dataset:
    """
    Parse the data files.

    Args:
        dataset_location: the location of the dataset
        bounding_box_mode: which strategy to use to feed the boundign boxes to the model
        normalize: normalize the bounding boxes

    Returns:
        dataset
    """
    data: Dataset = defaultdict(list)
    for subset in "testing", "training":
        filenames = glob.glob(f"{dataset_location}/{subset}_data/annotations/*")
        for filename in filenames:
            path = Path(filename)
            image_path = (path.parent.parent / "images" / path.stem).with_suffix(".png")
            image = Image.open(image_path).convert("RGB")
            json_data = read_json(filename)["form"]
            words: list[str] = []
            word_labels: list[int] = []
            bboxes: list[list[int]] = []
            annotations: dict[int, Annotation] = {}
            for id_, annotation in enumerate(json_data):
                assert id_ == annotation["id"]
                label = annotation["label"]
                entries = annotation["words"]
                linking = annotation["linking"]
                word_counter = 0
                for i, entry in enumerate(entries):
                    word = entry["text"]
                    # Skip empty strings
                    if word == "":
                        continue
                    if bounding_box_mode == "word":
                        bbox = normalize_bbox(
                            entry["box"], *image.size, normalize=normalize
                        )
                    elif bounding_box_mode == "block":
                        bbox = normalize_bbox(
                            annotation["box"], *image.size, normalize=normalize
                        )
                    else:
                        assert bounding_box_mode == "linking"
                        x1, y1, x2, y2 = annotation["box"]
                        for pair in linking:
                            for j in pair:
                                box = json_data[j]["box"]
                                x1 = min(x1, box[0])
                                y1 = min(y1, box[1])
                                x2 = max(x2, box[2])
                                y2 = max(y2, box[3])
                        bbox = normalize_bbox(
                            [x1, y1, x2, y2], *image.size, normalize=normalize
                        )

                    words.append(word)
                    bboxes.append(bbox)
                    word_label = (
                        "OTHER"
                        if label == "other"
                        else f"{'B' if word_counter == 0 else 'I'}-{label.upper()}"
                    )
                    word_labels.append(WORD_LABELS.index(word_label))
                    word_counter += 1
                text = annotation["text"]
                word_indices = list(range(len(words) - word_counter, len(words)))
                annotations[id_] = Annotation(
                    text=text, linking=linking, label=label, word_indices=word_indices
                )
            datum = Datum(
                basename=path.stem,
                image=image,
                annotations=annotations,
                words=words,
                word_labels=word_labels,
                bboxes=bboxes,
            )
            data[subset].append(datum)
    return data


def read_json(filename: str) -> dict[Any, Any]:
    """
    Parse json file.

    Args:
        filename: the json filename

    Return:
        parsed dictionary
    """
    with open(filename, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert isinstance(d, dict)
    return d


def normalize_bbox(
    bbox: list[int], width: int, height: int, normalize: bool = True
) -> list[int]:
    """
    Normalize the bounding box.

    Args:
        bbox: the bounding box
        width: the image width
        height: the image height
        normalize: if False, don't normalize

    Returns:
        normalize bounding box
    """
    if normalize is True:
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]
    return bbox
