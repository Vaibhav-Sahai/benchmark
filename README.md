## How to run BRIGHT w/ VLLM

### 1. Install requirements

```bash
conda create -n bright python=3.10
conda activate bright
pip install -r requirements.txt
```

### 2. .sh Script

```bash
cd BRIGHT
./single_task.sh [subtask] [model-name] [ft if finetuned model]
```
So ```./single_task.sh biology ../qwen2.5-7b-reasoning-merged/ ft``` will run finetuned model on the biology subtask with the model path being ```../qwen2.5-7b-reasoning-merged/```. You can substitute the model path with ```Qwen/Qwen2.5-7B```. 

So the command becomes 
```bash
./single_task.sh biology Qwen/Qwen2.5-7B 
```

Note: The .sh file has cuda visible set to 2 devices, change it as needed.

### 3. JSON Dir
The JSON Dir stores results inside of task specific subdir, the results json is called ```converted_results-ft.json``` if the model is finetuned and ```converted_results.json``` if the model is not finetuned.