"""
Evaluate model on test set.
"""

import argparse

import datasets
import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    PreTrainedModel,
)
from transformers.data.data_collator import default_data_collator

from .utils import WORD_LABELS, Datum, SampleMapping, parse_data, prepare_data

CLASS_LABELS = ["other", "question", "answer", "header"]


class Arguments(BaseModel):
    """
    Command-line arguments.

    Args:
        data: folder containing the training and testing data
        model: the model location to load
    """

    dataset: str
    model: str


def parse_args() -> Arguments:
    """
    Parse command-line arguments as an instance of `Arguments`.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Create and evaluate classifier.",
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Folder containing the training and testing data.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model location",
    )

    args = parser.parse_args()

    return Arguments(**vars(args))


def get_logits(model: PreTrainedModel, test_dataset: datasets.Dataset) -> list[Tensor]:
    """
    Get logits of all data points.

    Args:
        model: the model
        test_dataset: the test dataset

    Returns:
        logits
    """
    logits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for encoding in test_dataset:
        collated_encodings = {
            key: value.to(device)
            for key, value in default_data_collator([encoding]).items()
        }
        outputs = model(**collated_encodings)
        mask = collated_encodings["labels"][0] != -100
        logits.append(outputs["logits"][0][mask].detach().cpu())
        del collated_encodings, mask, outputs
    return logits


def load_model_and_processor(model_id: str) -> tuple[PreTrainedModel, AutoProcessor]:
    """
    Load the model and processor.

    Args:
        model_id: the model location

    Returns:
        the loaded model and processor
    """
    processor = AutoProcessor.from_pretrained(model_id, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_id).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model, processor


def post_process(
    logits: list[Tensor], sample_mapping: list[SampleMapping], data: list[Datum]
) -> tuple[list[str], list[str]]:
    """
    Get the true and predicted labels of all data points.

    Args:
        logits: the logits
        sample_mapping: mapping to recover the correct logits for an annotation
        data: the dataset

    Returns:
        true and predicted labels
    """
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    for j, item in enumerate(data):
        # Use index instead of id since some ids were removed
        for annotation in item.annotations.values():
            true_labels.append(annotation.label)
            word_logits = get_word_logits(
                j, annotation.word_indices, logits, sample_mapping
            )
            predicted_label = gather_predictions(word_logits)
            predicted_labels.append(predicted_label)
    return true_labels, predicted_labels


def gather_predictions(word_logits: Tensor) -> str:
    """
    Get a label prediction by adding up logits for all words.

    Args:
        word_logits: the per-word logits

    Returns:
        Predicted label
    """
    class_logits = [0.0, 0.0, 0.0, 0.0]
    for i, logit in enumerate(word_logits):
        class_logits[CLASS_LABELS.index("other")] += logit[
            WORD_LABELS.index("OTHER")
        ].item()
        for class_name in ("question", "answer", "header"):
            class_logits[CLASS_LABELS.index(class_name)] += logit[
                WORD_LABELS.index(f"{'B' if i == 0 else 'I'}-{class_name.upper()}")
            ].item()
    return CLASS_LABELS[np.argmax(class_logits)]


def get_word_logits(
    sample_index: int,
    word_indices: list[int],
    logits: list[Tensor],
    sample_mapping: list[SampleMapping],
) -> Tensor:
    """
    Get the logits for each of the given word indices.

    Args:
        sample_index: the sample index
        word_indices: the word indices
        logits: the logits
        sample_mapping: the sample mapping

    Returns:
        per-word logits
    """
    word_logits = torch.zeros((len(word_indices), 7))
    for i, mapping in enumerate(sample_mapping):
        if mapping.sample_index != sample_index:
            continue
        for j, word_index in enumerate(word_indices):
            if word_index < mapping.word_start_index:
                continue
            if word_index > mapping.word_end_index - 1:
                continue
            shifted_index = word_index - mapping.word_start_index
            word_logits[j] += logits[i][shifted_index]
    # Sanity check on indexing
    assert (torch.abs(word_logits) > 1e-9).all().item()
    return word_logits


def print_accuracy(true_labels: list[str], predicted_labels: list[str]) -> None:
    """
    Print the accuracy.

    Args:
        true_labels: the true labels
        predicted_labels: the predicted labels
    """
    n_total = len(true_labels)
    n_correct = sum(
        label1 == label2
        for label1, label2 in zip(true_labels, predicted_labels, strict=True)
    )
    print(f"Accuracy is {100 * n_correct / n_total:.1f}")


def main(args: Arguments) -> None:
    """
    Evaluate model.

    Args:
        args: the command-line arguments
    """
    data = parse_data(dataset_location=args.dataset)
    model, processor = load_model_and_processor(args.model)
    test_dataset, sample_mapping = prepare_data(data["testing"], processor)
    logits = get_logits(model, test_dataset)
    true_labels, predicted_labels = post_process(
        logits, sample_mapping, data["testing"]
    )
    print_accuracy(true_labels, predicted_labels)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
