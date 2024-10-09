max_new_tokens=$1
model_weights=$2

docker run -it \
  -v .:/data/shared/LLM/FastChat/temp_file \
  -v ${model_weights}:/data/shared/LLM/FastChat/models \
  --gpus "device=0" \
  csw_llm_deploy \
  sh -c "cd ./temp_file/ && \
         pip3 install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple && \
         cp qwen_related/utils.py /usr/local/lib/python3.8/dist-packages/transformers/generation/ && \
         cp qwen_related/modeling_qwen2.py /usr/local/lib/python3.8/dist-packages/transformers/models/qwen2/ && \
         python3 tf_test.py --max-new-tokens '${max_new_tokens}' --model fp16 > output/'fp16_memory_${max_new_tokens}'.log 2>&1 && \
         python3 tf_test.py --max-new-tokens '${max_new_tokens}' --model gptq4 > output/'gptq4_memory_${max_new_tokens}'.log 2>&1 && \
         python3 tf_test.py --max-new-tokens '${max_new_tokens}' --model awq > output/'awq_memory_${max_new_tokens}'.log 2>&1 && \
         python3 tf_test.py --max-new-tokens '${max_new_tokens}' --model fp16 --no-flash-attn > output/'fp16_memory_${max_new_tokens}_no_flash_attn'.log 2>&1 && \
         python3 extract.py --max-new-tokens '${max_new_tokens}'"