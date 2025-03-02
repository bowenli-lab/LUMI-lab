import pandas as pd
import numpy as np


N = 256 # 8 hrs

compute_failed_job_N = 15 # focus on failed jobs

virtual_lib_path="/home/pangkuan/projects/def-bowenli/pangkuan/data/15M_virtual_library.csv"

output_path = "/home/pangkuan/projects/def-bowenli/pangkuan/data/15m-lib/partitioned_txt"


if compute_failed_job_N:
    sub_N = 16
    output_path = f"/home/pangkuan/projects/def-bowenli/pangkuan/data/15m-lib/array-job-{compute_failed_job_N}"

df = pd.read_csv(virtual_lib_path)["combined_mol_SMILES"]
partitions = np.array_split(df, N)
for i, partition in enumerate(partitions):
    # save into txt
    print(f"processing {i} partition")
    if not compute_failed_job_N:
        partition.to_csv(f'{output_path}/{i}.txt', index=False, header=False)
    if i == compute_failed_job_N:
        sub_partitions = np.array_split(partition, sub_N)
        for j, sub_partition in enumerate(sub_partitions):
            sub_partition.to_csv(f'{output_path}/partitioned_txt/{j}.txt', index=False, header=False)
        break
        
