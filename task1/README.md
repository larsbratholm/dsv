# Task 1
## Usage
Training:
```
python -m task1.trainer data/ -o task1/model
```

Evaluation:
```
python -m task1.evaluate data/ task1/model
```
## Model description
I fine-tuned a linear classification layer of a LayoutLMv3 model for token classification.
Since the given dataset is used in the LayoutLMv3ForTokenClassification notebook,
I used some of the same data transformation strategies.
The block-level (per-id) labels were converted into word-level labels,
with different labels for the beginning and subsequent words in a block.

The only noticable change to the model and data architecture is that part of the input to the model
is a list of words and their corresponding bounding boxes.
The default usage is that the bounding box accompanying a word is the bounding box of the block the
word appears in.
I tested a word-level bounding box approach as well as using bounding boxes that encompasses the
bounding boxes of all the linked blocks.
The latter was an attempt at incorporating the linkage from the OCR.

Since the model provides word-level predictions, I added a post-processing step that
gathers the predictions to block-level (per-id).
This step could possibly introduce a bias, but it would require a lot more work to train
properly at the block-level.

## Data pre-processing
I removed empty string words as a pre-processing step.
Some of the examples had too many words for the model to support the input.
The LayoutLMv3 does not support the standard HuggingFace approaches of increasing this limit,
so I opted to split the input in two in the cases where this was an issue.
A better approach would be to create several sliding windows, but I just made two overlapping splits
when needed.

## Reults
The fitted model gives and accuracy around 83%

## Improvements
Other than the previously mentioned issues, the classifier can likely be improved by
adding a non-linearity (hidden layer).
Since I only fine-tune the linear classification layer on top of the untouched pre-trained model,
the model might be a bit too constrained, and it would be rather easy to change this.
Alternatively a second fine-tuning step could be added with low-rank adaptors of the entire model.
