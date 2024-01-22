# Instructions to Replicate CAI Results

Run the following command to generate 64 samples of the CAI dataset for debugging purposes. It should run pretty fast.

```
python examples/hh/generate_cai_dataset.py --push_to_hub
```

To generate the whole dataset, run the following command. It will take a while to run. Specify the number of instances to run in parallel with the `--instances` flag. You can also customize your own constitution file with the `--constitution_path` flag.

```
python examples/hh/generate_hh.py --push_to_hub --max_samples=-1 --instances=8 --constitution_path=examples/hh/constitution.json
```


An example generation is our pre-built CAI dataset

https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless


Here is a past run of the command:

```
(.venv) costa@login-node-1:/fsx/costa/tgi-swarm$ python examples/hh/generate_hh.py --push_to_hub --max_samples=-1 --instances=6
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Map: 100%|█████████████████████████████████████████████████████████| 42537/42537 [00:01<00:00, 32074.71 examples/s]
Map: 100%|███████████████████████████████████████████████████████████| 2312/2312 [00:00<00:00, 26755.85 examples/s]
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
running sbatch --parsable slurm/tgi_1705622089_tgi.slurm
Slurm Job ID: ['1188241', '1188242', '1188243', '1188244', '1188245', '1188246']
📖 Slurm Hosts Path: slurm/tgi_1705622089_host_tgi.txt
✅ Done! Waiting for 1188241 to be created                                                                         
✅ Done! Waiting for 1188242 to be created                                                                         
✅ Done! Waiting for 1188243 to be created                                                                         
✅ Done! Waiting for 1188244 to be created                                                                         
✅ Done! Waiting for 1188245 to be created                                                                         
✅ Done! Waiting for 1188246 to be created                                                                         
✅ Done! Waiting for slurm/tgi_1705622089_host_tgi.txt to be created                                               
obtained endpoints ['http://26.0.161.103:10613', 'http://26.0.173.246:38397', 'http://26.0.162.46:26736', 'http://26.0.161.138:49305', 'http://26.0.161.78:5552', 'http://26.0.161.153:30709']
⡿ Waiting for http://26.0.161.103:10613 to be reachable
Connected to http://26.0.161.103:10613
✅ Done! Waiting for http://26.0.161.103:10613 to be reachable                                                     
⣽ Waiting for http://26.0.173.246:38397 to be reachable
Connected to http://26.0.173.246:38397
✅ Done! Waiting for http://26.0.173.246:38397 to be reachable                                                     
⣟ Waiting for http://26.0.162.46:26736 to be reachable
Connected to http://26.0.162.46:26736
✅ Done! Waiting for http://26.0.162.46:26736 to be reachable                                                      
⣽ Waiting for http://26.0.161.138:49305 to be reachable
Connected to http://26.0.161.138:49305
✅ Done! Waiting for http://26.0.161.138:49305 to be reachable                                                     
⣽ Waiting for http://26.0.161.78:5552 to be reachable
Connected to http://26.0.161.78:5552
✅ Done! Waiting for http://26.0.161.78:5552 to be reachable                                                       
⣽ Waiting for http://26.0.161.153:30709 to be reachable
Connected to http://26.0.161.153:30709
✅ Done! Waiting for http://26.0.161.153:30709 to be reachable                                                     
Endpoints running properly: ['http://26.0.161.103:10613', 'http://26.0.173.246:38397', 'http://26.0.162.46:26736', 'http://26.0.161.138:49305', 'http://26.0.161.78:5552', 'http://26.0.161.153:30709']
✅ test generation
✅ test generation
✅ test generation
✅ test generation
✅ test generation
✅ test generation
running sudo docker run -d -p 60929:60929 --network host -v $(pwd)/slurm/tgi_1705622089_load_balancer.conf:/etc/nginx/nginx.conf nginx
running sudo docker logs 7c0bf1afddcaac304d3c1f104f8f66e787646561095ed27c851adf4a9afb5c2e
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
🔥 endpoint ready http://localhost:60929
WARNING: the first generation might hang a bit because of the multi-turn chat and long context.
 63%|███████████████████████████████████████████▊                          | 28054/44849 [1:24:25<12:58, 21.58it/s]
```

It finished after 1h37min. Note the tokens per second seems slow but it is because of the multi-turn chat and long context.

```
100%|██████████████████████████████████████████████████████████████████████| 44849/44849 [1:37:06<00:00,  7.70it/s]
Overall Tokens per Second: 2471.9039087752
```



## Advanced


You get generate your own system chats as follows, and you can customize the system chat according to your needs. If the computation is light, you can use the `--debug_endpoint="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"` flag to use the HF inference API, but note that it will be rate limited.


```
python examples/hh/generate_system_chat.py --debug_endpoint="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
```
