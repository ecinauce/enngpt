from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from pprint import pprint as pr


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
data_collator = DefaultDataCollator()

# model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
training_args = TrainingArguments(
    # output_dir="my_awesome_qa_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# trainer.train()

question = "Who is Jesus?"
# context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

with open("asv.txt", "r") as ctx:
    context = ctx.read()

# pr(context)

# question_answerer = pipeline("question-answering", model="my_awesome_distilbert_qa_model", tokenizer=tokenizer)
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer=tokenizer)
result = question_answerer(question=question, context=context)
print(result["answer"])