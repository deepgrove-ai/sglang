python -m sglang.launch_server   --model-path /scratch/ansh/models/maple_reference_model  --host 0.0.0.0   --port 30000   --mem-fraction-static 0.7   --trust-remote-code

export SGLANG_TORCH_PROFILER_DIR=/scratch/ansh/profiles/sglang
mkdir -p /scratch/ansh/profiles/sglang


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


python compare_tensors.py --plot --plot-out kl_plot.png --rtol 0.01 --atol 0.01 --tokenizer /scratch/ansh/models/maple_reference_model  | tee out_comp_with_plots.log

python plot_range.py --start-step 46 --end-step 52 --per-stage

python report_range.py --start-step 48 --end-step 49 --tokenizer-path /scratch/ansh/models/maple_reference_model