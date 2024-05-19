```bash
python batch_generator_finetune.py --input_files ./sample_data/calvin_scale_llama/train.jsonl --tokenizer ibm-granite/granite-3b-code-instruct --output_directory ./output/calvin/batches --file_prefix calvin --max_sequence_length 512 --batch_size 16
```