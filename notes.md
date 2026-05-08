python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model  --host 0.0.0.0   --port 30000   --mem-fraction-static 0.7   --trust-remote-code

python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model  --host 0.0.0.0   --port 30000   --mem-fraction-static 0.7   --trust-remote-code --disable-cuda-graph --disable-radix-cache --disable-hybrid-swa-memory --attention-backend fa3 --skip-0-warmup --tp 1  2>&1 | tee out_sglang_logs.log

export SGLANG_TORCH_PROFILER_DIR=/scratch/ansh/profiles/sglang
mkdir -p /scratch/ansh/profiles/sglang

and then to request: 

curl -X POST -H 'Content-Type: application/json' "http://0.0.0.0:30000/start_profile" -d '{"num_steps":3}'

curl -s http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/scratch/ansh/models/maple_reference_model", "prompt": "My name is", "temperature": 0, "max_tokens": 128}'


CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model  --host 0.0.0.0   --port 30000   --mem-fraction-static 0.7   --trust-remote-code  --disable-radix-cache --disable-hybrid-swa-memory --skip-server-warmup --disable-cuda-graph --attention-backend fa3 --enable-torch-compile --tp 1 --watchdog-timeout 3600  2>&1 | tee out_sglang_logs.log


## the command that got radix attention working: 

FA_DUMP=1 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model   --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 --trust-remote-code   --disable-radix-cache --disable-hybrid-swa-memory --skip-server-warmup   --disable-cuda-graph --attention-backend fa3 --enable-torch-compile  --tp 1   --watchdog-timeout 3600 2>&1 | tee out_sglang_logs.log

FA_DUMP=1 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model   --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 --trust-remote-code   --disable-radix-cache --disable-hybrid-swa-memory --skip-server-warmup   --disable-cuda-graph --attention-backend fa3 --enable-torch-compile  --tp 1   --watchdog-timeout 3600 2>&1 | tee out_sglang_logs.log

### this also works 

*you can either have cuda graphs or torch compile*

**torch compile** 
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model   --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 --trust-remote-code --attention-backend fa3 --enable-torch-compile --tp 1 --disable-cuda-graph  --watchdog-timeout 3600 2>&1 | tee out_sglang_logs.log

**cuda graphs**  (failing for some reason??!)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model   --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 --trust-remote-code --attention-backend fa3 --tp 1 --cuda-graph-max-bs 16  --watchdog-timeout 3600 2>&1 | tee out_sglang_logs.log


CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model   --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 --trust-remote-code --attention-backend fa3 --tp 1 --disable-cuda-graph  --watchdog-timeout 3600 2>&1 | tee out_sglang_logs.log

# with torch compile 

python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model  --host 0.0.0.0   --port 30000   --mem-fraction-static 0.7   --trust-remote-code  --disable-radix-cache --disable-hybrid-swa-memory --attention-backend maple_fa --skip-server-warmup --disable-cuda-graph --enable-torch-compile --tp 1  2>&1 | tee out_sglang_logs.log

## prefill

export SGLANG_TORCH_PROFILER_DIR=/scratch/ansh/profiles/sglang

python -m sglang.bench_serving \
  --backend sglang \
  --model /scratch/ansh/models/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 20 \
  --profile

## decode

python -m sglang.bench_serving \
  --backend sglang \
  --model /scratch/ansh/models/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 256 \
  --random-range-ratio 1 \
  --num-prompts 20 \
  --profile

python compare_inference.py

python compare_tensors.py --plot --plot-out kl_plot.png --rtol 0.01 --atol 0.01 --tokenizer /scratch/ansh/models/maple_reference_model  | tee out_comp_with_plots.log

python plot_range.py --start-step 46 --end-step 52 --per-stage

python report_range.py --start-step 48 --end-step 49 --tokenizer-path /scratch/ansh/models/maple_reference_model