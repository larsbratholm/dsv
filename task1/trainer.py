"""
Train classifier.
"""

import argparse

import datasets
import torch
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoProcessor,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator

from .utils import WORD_LABELS, parse_data, prepare_data


class Arguments(BaseModel):
    """
    Command-line arguments.

    Args:
        data: folder containing the training and testing data
        output_folder: where to store the finetuned model
        early_stopping_patience: the early stopping patience
        learning_rate: the learning rate
        max_steps: the maximum number of training steps
        batch_size: the batch size
        logging_steps: the number of steps between evaluation and checkpointing
    """

    dataset: str
    output_folder: str
    early_stopping_patience: int
    learning_rate: float
    max_steps: int
    batch_size: int
    logging_steps: int


def parse_args() -> Arguments:
    """
    Parse command-line arguments as an instance of `Arguments`.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train classifier.",
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Folder containing the training and testing data.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="Where to store the finetuned model",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="The early stopping patience.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="The learning rate.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="The maximum number of training steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="The number of steps between evaluation and checkpointing.",
    )

    args = parser.parse_args()

    return Arguments(**vars(args))


def prepare_model(model: PreTrainedModel) -> None:
    """
    Train all parameters.

    Args:
        model: the model
    """
    for param in model.parameters():
        param.requires_grad_(True)
    # model.classifier.weight.requires_grad_(True)
    # model.classifier.bias.requires_grad_(True)


def fit_model(
    args: Arguments,
    model: PreTrainedModel,
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
) -> None:
    """
    Fit the model.

    Args:
        args: the command-line arguments
        model: the model
        train_dataset: the training dataset
        eval_dataset: the evaluation dataset
    """
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    ]
    training_arguments = TrainingArguments(
        output_dir=args.output_folder,
        eval_strategy="steps",
        eval_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_strategy="steps",
        save_steps=args.logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


def load_model_and_processor(
    model_id: str = "microsoft/layoutlmv3-base",
) -> tuple[PreTrainedModel, AutoProcessor]:
    """
    Load the pre-trained model and processor.

    Args:
        model_id: the model to load

    Returns:
        the loaded model and processor
    """
    num_labels = len(WORD_LABELS)

    processor = AutoProcessor.from_pretrained(model_id, apply_ocr=False)
    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label={key: value for key, value in enumerate(WORD_LABELS)},
        label2id={value: key for key, value in enumerate(WORD_LABELS)},
    )
    model = AutoModelForTokenClassification.from_pretrained(model_id, config=config).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model, processor


def save_model(
    model: PreTrainedModel, processor: AutoProcessor, output_folder: str
) -> None:
    """
    Save the model and processor.

    Args:
        model: the model
        processor: the processor
        output_folder: the output folder
    """
    model.save_pretrained(output_folder)
    processor.save_pretrained(output_folder)


def main(args: Arguments) -> None:
    """
    Finetune and save model.

    Args:
        args: the command-line arguments
    """
    data = parse_data(dataset_location=args.dataset)
    model, processor = load_model_and_processor()
    prepare_model(model)
    train_dataset = prepare_data(data["training"], processor)[0]
    eval_dataset = prepare_data(data["testing"], processor)[0]
    fit_model(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    save_model(model, processor, output_folder=args.output_folder)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
