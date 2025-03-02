import modal
from modal import Secret, App, Function


docker_hub_secret = Secret.from_name("docker-secret")
image = modal.Image.from_registry("pangkuandocker/sdl-infer:0.0.2",
                                  secret=docker_hub_secret,)
app = App(image=image)

# 16G to mb 
mem_request = 16 * 1024
@app.function(gpu=modal.gpu.A100(size="40GB"),
              cpu=4, 
              memory=mem_request,
              enable_memory_snapshot=False,
              retries=3)
def infer(shard_id, num_shards, model_name):
    print(f"processing shard_id: {shard_id}, num_shards:{num_shards}, model_name: {model_name}")
    
    import os
    # print(os.environ)
    import subprocess
    import pandas as pd

    backbone_path = "backbone_place_holder.pt"
    dict_path = "./dict.txt"
    pretrained_weight = model_name
    lmdb_dir = "all-lmdb"
    result_csv = "test_result.csv"
    eval_batch_size = 64
    smi_path = "smi_name.txt"

    env = {}
    env.update(os.environ)

    def _exec_subprocess(cmd, env):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                # print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    os.chdir("/app")

    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
    _exec_subprocess(cmd, env)

    cmd = [
        "python", 
        "./unimol_inference.py",
        f"--backbone-path={backbone_path}",
        f"--dict-path={dict_path}",
        f"--pretrained-weight={pretrained_weight}",
        f"--lmdb-dir={lmdb_dir}",
        f"--result-csv={result_csv}",
        f"--eval-batch-size={str(eval_batch_size)}",
        f"--smi-path={smi_path}",
        f"--shard-id={str(shard_id)}",
        f"--num-shards={str(num_shards)}"
    ]
    _exec_subprocess(cmd, env)

    output = pd.read_csv(result_csv)
    print(f"partition {shard_id+1}/{num_shards} done: {len(output)} rows.")
    return output.to_dict()

@app.local_entrypoint()
def main():
    N = 10
    for fold in range(1, 6):
        print(f"Processing fold {fold}")
        model_name = f"KuanP/a100-contrastive-ckpt-2_2024-07-09_fold_{fold}"
        
        input_tuple_list = []
        for name in [model_name]:
            for i in range(N):
                input_tuple_list.append((i, N, name))
        
        results = [*infer.starmap(input_tuple_list)]

        import pandas as pd
        
        # save the results to pkl
        with open(f"final_result_dict_list_{fold}.pkl", "wb") as f:
            import pickle
            pickle.dump(results, f)

        df_list = [pd.DataFrame.from_dict(result) for result in results]
        df = pd.concat(df_list)
        df.to_csv(f"final_result_{fold}.csv", index=False)
        # get result length
        for i, result in enumerate(df_list):
            print(f"Local: partition {i+1}/{N} done: {len(result)} rows.")
        