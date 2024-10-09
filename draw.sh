docker run -it \
  -v .:/data/shared/LLM/FastChat/temp_file \
  csw_llm_deploy \
  sh -c "cd ./temp_file/ && \
         pip3 install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple && \
         python3 draw_steps_concatenated.py && \
         python3 draw_steps_stacked.py && \
         python3 draw_layers.py"