# Task 3 (Case 1)
## Model considerations
A [LayoutLM](https://huggingface.co/docs/transformers/v4.48.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) variant for question answering will probably work well for the task, assuming that (some) of the document types are image-based, and the computational cost is manageable.
If not most of the below discussion and considerations is not exclusive to an multi-modal approach, and a text-only model could be used.
Document types like emails are often text-only, and could be treated with a separate model, or by conversion to image.

In terms of experiments regarding fine-tuning, the effort required will be related to how expressive the pre-trained model is, which will consist of some trial and error.
Ideally one can construct some metric of target accuracy before the experiments to have an idea of what is good enough.

I would suggest the following experiments regarding the overall approach (data preparation, output formatting and fine-tuning is implied):

### 1.
The model might be expressive enough that you can just formulate the question and give the input image.

### 2.
The model is having issues due to the varying document types, which could express itself in bad performance on specific types or tasks.
One approach would be to include custom tags in the input that helps narrow down the subproblem (e.g. document type or subtask).
This would add in a bit more complexity with regards to labeling during training and at runtime.
However, it should be fine to just label a subset and otherwise use masking during training, and have the model predict labels at runtime if needed.
The input could be something similar to:
```
[DOCUMENT_CLASSIFICATION]Is the document an invoice, customs document, or email?
[INVOICE]Extract purchase-related fields from the invoice.
```

The theory behind this approach is that the attention heads will focus in on specific document types or tasks during fine-tuning.
Given that new tags are added, a pre-training step is required that masks any old tokens, to get a good starting point.

### 3.
The answering decoder can be duplicated and specialized for different tasks, while keeping a shared base model, similar to mixture of experts approaches.
The simplest approach is similar to above, where there's a hard classifier that determines which decoder is used to produce the response, but some other gating mechanism would work as well.
Depending on the exact architecture, training could be done in stages, where parts of the model is training on subsets of the data, followed by a final joint finetuning.


## Data
One can likely get enough training data from publically available sources, especially since it is a relatively standard task.
Some manually curated examples would be helpful for testing, and ideally one would implement a method for the users to submit samples that the deployed model handled incorrectly for incremental improvements.
