from datasets import Dataset
import os
from transformers import TrainingArguments, Trainer
import transformers
from hf_unimol import UniMolConfig, init_unimol_backbone, UniMol
from lmdb_dataset import load_dataset, collate_fn
from hg_mterics import compute_metrics
from fused_adam import FusedAdam
import wandb
from datetime import datetime
import numpy as np
import torch


# ===== config =====
os.environ["WANDB_PROJECT"] = "1920-model"  # name your W&B project
weight_path = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/mol_pre_no_h_220816.pt"
dict_path = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/dict.txt"
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
model_config = UniMolConfig(
    input_dim=512,
    inner_dim=512,
    num_classes=1,
    dropout=0,
    decoder_type="mlp",
)
run_name = f"4CR_customized_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
epoch_num = 24
warmup_ratio = 0.06
output_path = "/home/pangkuan/dev/dd_1/uni-mol-model/results"

# load model and dictionary
model_backbone, dictionary = init_unimol_backbone(weight_path, dict_path=dict_path)
model = UniMol(model_backbone, model_config, dictionary)

# build huggingface dataset from lmdb
train_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/train.lmdb",
    "train",
)
valid_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/valid.lmdb",
    "valid",
)
test_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/test.lmdb",
    "test",
)


hf_train_data = Dataset.from_generator(
    lambda: train_data,
)

hf_valid_data = Dataset.from_generator(
    lambda: valid_data,
)

hf_test_data = Dataset.from_generator(
    lambda: test_data,
)
wandb.log(
    {
        "weight_path": weight_path,
    }
)

training_steps = len(hf_train_data) * epoch_num
warmup_steps = int(training_steps * warmup_ratio)

optimizer = FusedAdam(
    model.parameters(),
    lr=1e-4,
    eps=1e-6,
    betas=(0.9, 0.99),
)

scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps,
)

training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=epoch_num,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    logging_dir="./logs",
    fp16=True,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    report_to="wandb",
    run_name=run_name,
    label_names=["target", "smi_name"],
    load_best_model_at_end=True,
    optim="adamw_torch",
    metric_for_best_model="relaxed_spearman",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_data,
    eval_dataset=hf_valid_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=None,
    optimizers=(optimizer, scheduler),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
output = trainer.predict(
    hf_test_data,
    metric_key_prefix="test",
)
print(output.metrics)
wandb.log(output.metrics)


trainer.save_model(f"/home/pangkuan/dev/dd_1/uni-mol-model/{run_name}")

wandb.finish()
