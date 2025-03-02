from pathlib import Path
import pandas as pd
from transformers import Trainer, TrainingArguments
import sys
import argparse
sys.path.insert(0, "../../")
from unimol.hf_unimol import UniMol, UniMolConfig, init_unimol_backbone
from datasets import Dataset
from unimol.lmdb_dataset import collate_fn, load_dataset, SliceDataset
import numpy as np

def main(backbone_weight_path, dict_path, pretrained_weight, 
         lmdb_dir, result_csv, eval_batch_size, smi_path,
         shard_id, num_shards, return_representation=False, load_from_local=False):
    # validate the shard_id and num_shards
    if shard_id != -1 and num_shards != -1:
        assert shard_id >= 0 and shard_id < num_shards
        print(f"Processing shard {shard_id+1} out of {num_shards}")
    if (shard_id == -1 and num_shards != -1) or (shard_id != -1 and num_shards == -1):
        raise ValueError("Both shard_id and num_shards must be provided")
        
    
    # Initialize the model backbone and dictionary
    model_backbone, dictionary = init_unimol_backbone(backbone_weight_path, dict_path=dict_path)

    # Create the model configuration
    model_config = UniMolConfig(
        input_dim=512,
        inner_dim=512,
        num_classes=1,
        dropout=0,
        decoder_type="mlp",
        return_representation=return_representation,
    )

    # Load the saved model
    model = UniMol.from_pretrained(pretrained_weight, config=model_config)
    if load_from_local:
        model.init_backbone(model_backbone)

    # Prepare the test dataset
    lmdb_dir = Path(lmdb_dir)
    test_data = load_dataset(dictionary, str(lmdb_dir / "test.lmdb"), "test")
    hf_test_data = test_data
    print("len(hf_test_data):", len(hf_test_data))
    
    
    smi_name_list = []
    with open(smi_path, "r") as f:
        smi_name_list = f.readlines()
        smi_name_list = [smi_name.strip() for smi_name in smi_name_list]
    
    # calculate shard size
    if shard_id != -1 and num_shards != -1:
        shard_size = len(hf_test_data) // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size
        if shard_id == num_shards - 1:
            end_idx = len(hf_test_data)
        hf_test_data = SliceDataset(hf_test_data, start_idx, end_idx)
        print(f"Shard size: {shard_size}, start_idx: {start_idx}, end_idx: {end_idx}")
        smi_name_list = smi_name_list[start_idx:end_idx]
        
    assert len(hf_test_data) == len(smi_name_list), "Length of hf_test_data and smi_name_list must be the same."

    # Define the trainer for inference
    training_args = TrainingArguments(
        output_dir=pretrained_weight,
        per_device_eval_batch_size=eval_batch_size,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        eval_accumulation_steps=512,
        label_names=["target","smi_name",],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
    )

    # Run inference on the test dataset
    test_output = trainer.predict(hf_test_data)


    pred_result = test_output.predictions[0].squeeze().tolist()
    
    print("len(pred_result):", len(pred_result))
    print("len(smi_name_list):", len(smi_name_list))
    assert len(pred_result) == len(smi_name_list), "Length of pred_result and smi_name_list must be the same."

    df = pd.DataFrame({"smi_name": smi_name_list, 
                       "pred_result": pred_result})


    # Save the results to CSV
    df.to_csv(result_csv, index=False)
    
    if return_representation:
        representation = test_output.predictions[1].squeeze()
        # make it fp16 to save space
        representation = representation.astype("float16")
        #  representation to npy
        dest_name = result_csv.replace(".csv", "_representation")
        np.savez_compressed(dest_name, representation)
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UniMol inference on a test dataset.")
    parser.add_argument("--backbone-path", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--dict-path", type=str, required=True, help="Path to the dictionary file.")
    parser.add_argument("--pretrained-weight", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--lmdb-dir", type=str, required=True, help="Path to the LMDB directory.")
    parser.add_argument("--result-csv", type=str, required=True, help="Path to save the results CSV.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--smi-path", type=str, required=True, help="Path to the precomputed SMILES file.")
    parser.add_argument("--shard-id", type=int, default=-1, help="Shard ID for distributed training.")
    parser.add_argument("--num-shards", type=int, default=-1, help="Number of shards for distributed training.")
    parser.add_argument("--return-representation", action="store_true", help="Return the representation.")
    parser.add_argument("--load-from-local", action="store_true", help="Load the backbone from local.")

    args = parser.parse_args()
    main(args.backbone_path, 
         args.dict_path, 
         args.pretrained_weight, 
         args.lmdb_dir,
         args.result_csv,
         args.eval_batch_size,
         args.smi_path,
         args.shard_id, 
         args.num_shards,
         args.return_representation)

