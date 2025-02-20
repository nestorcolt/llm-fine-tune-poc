from transformers import (DataCollatorForLanguageModeling,
                          AutoModelForCausalLM,
                          PhiForCausalLM,
                          BitsAndBytesConfig,
                          AutoTokenizer,
                          TrainingArguments,
                          set_seed,
                          Trainer,
                          logging)

from peft import LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
from functools import partial
from peft import PeftModel
import pandas as pd
import numpy as np
import evaluate
import torch
import time
import os

load_dotenv()
logging.set_verbosity_info()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"


##############################################################################################################

def timeit(func):
    """
    A decorator that times the execution of the function it decorates.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the decorated function
        end_time = time.time()  # Capture the end time
        print(f"\n\nFunction {func.__name__} took {end_time - start_time} seconds to execute.")
        return result

    return wrapper


##############################################################################################################


def get_tokenizer(model_name, **kwargs):
    # Now, let’s configure the tokenizer, incorporating left-padding to optimize memory usage during training.
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_nameuse_fast=True, kwargs=kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


@timeit
def run_model_prompt(input_model, tokenizer, processor="cuda", index=10, peft=False):
    seed = 42
    set_seed(seed)

    prompt = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    formatted_prompt = f"Instruction: Summarize the following conversation.\nPrompt: {prompt}"

    inputs = tokenizer(formatted_prompt,
                       return_tensors="pt").to(processor)

    response = input_model.generate(**inputs, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2)

    output = tokenizer.decode(response[0], skip_special_tokens=True).split("Output:")[-1]
    prefix, success, result = output.partition('###')

    # Print the results
    print("--------------------------------------------------------------------------------------------------------")
    print(f"INPUT PROMPT:\n{prompt}")
    print()
    print("--------------------------------------------------------------------------------------------------------")
    print(f"BASELINE HUMAN SUMMARY:\n{summary}")
    print()
    print("--------------------------------------------------------------------------------------------------------")

    if peft:
        print(f'PEFT MODEL SUMMARY:{prefix}')
    else:
        print(f'BASE MODEL SUMMARY:\n{prefix}')


##############################################################################################################
# Preprocessing the dataset
def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['dialogue']}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample['summary']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)

        if max_length:
            print(f"Found max lenth: {max_length}")
            break

    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")

    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    return dataset.shuffle(seed=seed)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_number_of_trainable_model_parameters(model):
    """
    This function prints the number of trainable model parameters, total model parameters,
    and the percentage of the trainable model parameters.

    Args:
    peft_model (object): A model object that has attributes for trainable and total parameters.

    Returns:
    None: This function prints the values and does not return anything.
    """

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = count_trainable_parameters(model)

    # Calculate the percentage of trainable parameters
    percentage_trainable = (trainable_parameters / total_parameters) * 100

    # Print the required information
    print(f"all model parameters: {total_parameters}")
    print(f"trainable model parameters: {trainable_parameters}")
    print(f"percentage of trainable model parameters: {percentage_trainable:.2f}%")


##############################################################################################################
def pre_process_data(tokenizer, dataset, target_model):
    # seed for dataset
    seed = 42
    set_seed(seed)

    # Get the maximum length of the model
    max_length = get_max_length(target_model)

    train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['train'])
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['validation'])

    # After loading the dataset
    print(f"Dataset loaded. Size: {len(dataset)}")

    # To inspect a few samples before preprocessing
    print(f"Sample data before preprocessing: {dataset['train'][:3]}")

    # After preprocessing, assuming 'pre_process_data' returns processed datasets
    print(f"Train dataset size after preprocessing: {len(train_dataset)}")
    print(f"Eval dataset size after preprocessing: {len(eval_dataset)}")

    # To inspect a few samples after preprocessing
    print(f"Sample train data after preprocessing: {train_dataset}")
    print(f"Sample eval data after preprocessing: {eval_dataset}")

    return train_dataset, eval_dataset


##############################################################################################################
def train_peft_adapter(peft_model, train_dataset, eval_dataset, tokenizer):
    output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

    peft_training_args = TrainingArguments(
        warmup_steps=1,
        max_steps=1000,
        save_steps=25,
        eval_steps=25,
        logging_steps=25,
        full_determinism=True,
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        evaluation_strategy="steps",
        do_eval=True,
        gradient_checkpointing=True,
        report_to=None,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        overwrite_output_dir=True,
        group_by_length=True,
    )

    peft_model.config.use_cache = False

    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    return peft_trainer


##############################################################################################################
# Parameters
login(token=os.getenv('HF_TOKEN'))

test_result_model = False
test_base_model = False
evaluate_rough = True
train_model = False

# Let’s now load Phi-2 using 4-bit quantization from HuggingFace.
model_name = 'microsoft/phi-2'
device_map = {"": 0}

huggingface_dataset_name = "neil-code/dialogsum-test"
dataset = load_dataset(huggingface_dataset_name)

# Create Bitsandbytes configuration
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

if train_model:
    original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          token=True)

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=16,  # Rank
        lora_alpha=16,
        target_modules=[

            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()

    # Get the number of trainable parameters and the total number of parameters
    peft_model = original_model.get_peft_model(original_model, config)
    print_number_of_trainable_model_parameters(peft_model)

    # Get the tokenizer
    tokenizer = get_tokenizer(model_name,
                              trust_remote_code=True,
                              padding_side="left",
                              add_eos_token=True,
                              add_bos_token=True)

    # training layer
    train_dataset, eval_dataset = pre_process_data(tokenizer, dataset, target_model=original_model)
    peft_trainer = train_peft_adapter(peft_model, train_dataset, eval_dataset, tokenizer)
    peft_trainer.train()

if test_base_model:
    original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          token=True)

    # Get the tokenizer
    tokenizer = get_tokenizer(model_name,
                              trust_remote_code=True,
                              padding_side="left",
                              add_eos_token=True,
                              add_bos_token=True)

    # inference
    run_model_prompt(original_model, tokenizer, processor="cpu")

if test_result_model:
    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      token=True)

    checkpoint_path = "/home/ec2-user/pycharm_project_132/peft-dialogue-summary-training-1708173843/checkpoint-1000"
    ft_model = PeftModel.from_pretrained(base_model,
                                         checkpoint_path,
                                         torch_dtype=torch.float16,
                                         is_trainable=False)

    # Get the tokenizer
    tokenizer = get_tokenizer(model_name,
                              add_bos_token=True,
                              trust_remote_code=True)

    # process data
    train_dataset, eval_dataset = pre_process_data(tokenizer, dataset, target_model=base_model)

    # inference
    run_model_prompt(ft_model, tokenizer, processor="cuda", index=10, peft=True)

if evaluate_rough:
    original_model = PhiForCausalLM.from_pretrained(model_name,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    trust_remote_code=True,
                                                    token=True)

    checkpoint_path = "/home/ec2-user/pycharm_project_132/peft-dialogue-summary-training-1708173843/checkpoint-1000"
    ft_model = PeftModel.from_pretrained(original_model,
                                         checkpoint_path,
                                         torch_dtype=torch.float16,
                                         is_trainable=False)
    index_examples = 10
    dialogues = dataset['test'][0:index_examples]['dialogue']
    human_baseline_summaries = dataset['test'][0:index_examples]['summary']

    original_model_summaries = []
    instruct_model_summaries = []
    peft_model_summaries = []

    # Get the tokenizer
    tokenizer = get_tokenizer(model_name,
                              add_bos_token=True,
                              trust_remote_code=True)

    for idx, dialogue in enumerate(dialogues):
        prompt = f"Instruction: Summarize the following conversation.\nPrompt: {dialogue}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # original
        response = original_model.generate(**inputs,
                                           max_length=1024,
                                           num_return_sequences=1,
                                           no_repeat_ngram_size=2)

        output = tokenizer.decode(response[0], skip_special_tokens=True).split("Output:")[-1]
        original_model_text_output, success, result = output.partition('###')

        # peft
        response = ft_model.generate(**inputs,
                                     max_length=1024,
                                     num_return_sequences=1,
                                     no_repeat_ngram_size=2)

        output = tokenizer.decode(response[0], skip_special_tokens=True).split("Output:")[-1]
        peft_model_text_output, success, result = output.partition('###')

        original_model_summaries.append(original_model_text_output)
        peft_model_summaries.append(peft_model_text_output)

    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
    df = pd.DataFrame(zipped_summaries,
                      columns=['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])

    print(f"DATAFRAME:\n{df}\n")

    # Evaluate the model
    print("Evaluating the model using ROUGE metrics ----------------------------------------------------------")
    rouge = evaluate.load('rouge')

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('PEFT MODEL:')
    print(peft_model_results)

    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))

    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value * 100:.2f}%')
