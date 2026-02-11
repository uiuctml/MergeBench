CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.run --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train llama-3.1-8b_code.yaml --deepspeed zero3.json
