
# 1. CgmGPT: causal language modeling on Continuous Glucose Monitor Data

The GPT Causal Language Model Trained on Continuous Glucose Monitor (CGM) on both Type-1 and Type-2 patients. 

# 2. Repo Structure

## a. `run_case_scope_whole`

This python script takes the patient $p_i$'s CGM record set $R_i^{cgm}$ as the trigger records, and uses them to the scope case $C_i^{scope}$. 

> Suppose a patient have 100 CGM record, then we have 100 cases $c_{ij} = (p_i, t_{ij})$ from $C_i^{scope}$, where $p_i$ is the patient and $t_{ij}$ is the observation time.

At the same time, for the patient $p_i$, the script `run_case_scope_whole.py` will classify him/her into a group based on $p_i$'s `Cohort`, `DiseaseType`, `Gender`, `YearOfBirth`. There `DiseaseType` is either type-1 diabetes or type-2 diabetes. 

**Usage**:
```shell
python run_case_scope_whole.py --record_name CGM5Min
```

You can find the scope case $C^{scope}$ for all patients in the folder: 
 `Data_CGMGPT/CaseFolder/{groupid}_{groupname}_whole.p`. 

**Notebook**:

The notebook `notebook/a-run_case_scope_whole.ipynb` is the notebook to develop this script. 




## b. `run_caseobs_recckpd_tkn`

### b.1 description

The python script `run_caseobs_recckpd_tkn.py` get the case level observation to a case $c_{ij} = (p_i, t_{ij})$. 


In mathematics, we have a case-level feature to calculate, and call it `CaseToken`. Then it can be noted as case level feature function $\Phi_{casetkn}$ , that $a_{ij} = \Phi_{casetkn}(c_{ij}, R_i)$. 

* $c_{ij}$ is the case of $(p_i, t_{ij})$
* $R_{i}$ is the patient $p_i$'s record set. $R_{i} = \cup R_i^{recname}$, where $recname$ is the name for different record types.
* $\Phi_{casetkn}$ is the function to get the case level features $a_{ij}$ to case $c_{ij}$ at the observation time $t_{ij}$, based on the patient $p_i$'s record set $R_i$. 

### b.2 Types of $a_{ij}$ and $\Phi$
Only subsets of $R_{i}$ will be used to calculate $a_{ij}$. There are different types of $a_{ij}$ and $\Phi_{casetkn}$ based on the subset of $R_i$ used to calculate the features.
* Standing at the observation time of $t_{ij}$ for case $c_{ij}$, the records happened before $t_{ij}$ is the before-record set: $R_i^{bf}$, and records happened after $t_{ij}$ is the after-record set: $R_i^{af}$. 

* If $R_i^{bf}$ is the input to $\Phi_{casetkn}$, the returned feature $a_{ij}$ will be used as the input features $x_{ij}$.
* If $R_i^{af}$ is the input to $\Phi_{casetkn}$, the returned feature $a_{ij}$ will be used as the future outcome label $y_{ij}$. 

* Only the case with both $x_{ij}$ and $y_{ij}$ can because an AI model development point: $(x_{ij}, y_{ij}) \in C_i^{dev}$. 

* We use the `CheckPeriod` $ckpd_{ij}$ anchored in $t_{ij}$ to select $R_i^{ckdp_{ij}}$ from $R_i$. The $ckpd$ can be `Bf2M`, `Bf24H`, `Af2H`, etc.

### b.3 Record Observation (Ckpd, RecName, Field)
To prepare inputs to $\Phi$ at the $c_{ij}$'s observation time $t_{ij}$, we can get an observation of RecName-CheckPeriod-Field: $(name, ckpd, \phi_{name, fld})$ from $R_i$, where $\phi_{name, fld}$ is record-level feature function. 

* `CheckPeriod`: The check period $ckpd_{ij}$ anchored with observation time $t_{ij}$ in the case $c_{ij}$. The options can be `Bf24H`, `Af2H` etc. 

* `RecName`: $name$ for $R_{i}^{name}$, like `CGM5Min`, `FoodRec` (in the future). Together with `CheckPeriod`, we have $R_{i}^{ckpd_{ij}, name}$.

* `Field` (Optional): The record-level feature function $\phi_{name, fld}$ for the field $fld$. We have $z_k = \phi_{name, fld}(r_k)$, where $r_{k} \in R_{i}^{ckpd_{ij}, name}$. Then we have a record observation: $recobs_i = R_{i}^{ckpd_{ij}, name, \phi_{name, fld}}$

For one case-level function $\Phi$, its inputs can be multiple observation tuples $(name, ckpd, \phi_{name, fld})$. These observations will be processed to case-level features $a_{ij}$. 

### b4. Case Obseravtion

**case_tkn**

There are different types of $\Phi$. For each $\Phi$, we will write the funtion tools and save them into the module `fn_casetkn`. For example: `1TknIn5Min`, `RecNum`, etc. 

### b5. Code

**Usage:**
```shell
python run_caseobs_recckpd_tkn.py \
    --group_id 0 \
    --scope_type "whole" \
    --rec_ckpd_list "Bf24H-CGM5Min" \
    # --record_observations "Bf24H-CGM5Min" \
    --case_tkn "RecNum"
```

```shell
python run_caseobs_recckpd_tkn.py \
    --group_id 1 \
    --case_type "dev" \
    --record_observations 'Bf24H-CGM5Min-N2Cin1' \
    --sfx_list 'tknidx' \
    --case_tkn '1TknIn5Min' \
    --batch_size 500 
```

`--group_id`: the id for a group of patients.

`--scope_type`: the scope cases type. `whole` means all scope cases. We will have different versions of scope case, i.e., `use`, `label`, `dev`, etc. 

`--rec_ckpd_list`: the observation of RecCkpd $(RecName, Ckpd)$. It can be `CGM5Min-Bf24H`, or `CGM5Min-Af2H`. Or we call it `--rec_obs`

`--case_tkn`: the casetkn function $\Phi$. Here the function $\Phi$ is `RecNum`, which return the record number of the observation $(RecName, Ckpd)$.


You can find the case-level features $a_{ij}$ for all patients in the folder: 
`Data_CGMGPT/CaseObserver/{groupid}_{groupname}_{scope_type}/{&RecObs}_{&Fld}_{CaseTkn}{tkn/tknidx/wgt}_size{CaseNum}`. 

Serveral Examples: 
* `{ro:Af2H-CGM5Min-N2Cin1}_{ct:1TknIn5Min}{tknidx}_size{1000}`
* `{ro:Af1W-EgmEdu}_{ct:FutEdu}{tknwgt}_size{1000}`
* `{ro:Af2M10D-WeightU&Bf1D-WeightU&PHeight}_{ct:FutWeight}{tknwgt}_size{1000}`


**Tokenizer**:

Tokenizer will also be generated based on $Field$ and $\Phi_{casetkn}$. 


**Notebook**:

The notebook `notebook/b-run_caseobs_recckpd_tkn.ipynb` is the notebook to develop this script. 


## c. `run_case_scope_filter`


With different case observations $co$, you can have different features: $a_{ij}$.

These case observation feature $a_{ij}$ can be used for different purposes.

- Filtering
- Feature Engineering. 


```shell
python run_case_scope_filter.py \
    --group_id 1 \
    --scope_type whole \
    --recckpd_casetkn_observer_list "CGM5Min-Bf24H-RecNum" "CGM5Min-Af2H-RecNum" \
    --case_filter sftflt
```

```shell
python run_case_scope_filter.py \
    --group_id 1 \
    --case_type whole \
    --case_observations "CGM5Min-Bf24H-RecNum" "CGM5Min-Af2H-RecNum" \
    --case_filter dev
```

```powershell
# You can ask GPT to change this to shell version.
python .\run_case_split_aidataset.py `
    --group_id 1 `
    --scope_type 'sftflt' `
    --task "CGMGPT" `
    --case_tkn_name_list 'BfCGM:CGM5Min*Bf24H*N2Cin1*tknidx' 'AfCGM:CGM5Min*Af2H*N2Cin1*tknidx' `
    --downsample_ratio 0.1 `
    --out_ratio 0.1 `
    --test_ratio 0.1 `
    --valid_ratio 0.1
```

```powershell
# You can ask GPT to change this to shell version.
python .\run_case_split_aidataset.py `
    --group_id 1 `
    --scope_type 'dev' `
    # --case_observations 'BfCGM:Bf24*CGM5Min*tknidx' 'AfCGM:CGM5Min*Af2H*N2Cin1*tknidx' `
    --downsample_ratio 0.1 ` # this should moved to case-filter
    --out_ratio 0.1 `
    --test_ratio 0.1 `
    --valid_ratio 0.1
```





# 3. Prepare Case Observer Data

## a. run case whole scope

```shell
python run_case_scope_whole.py --record_name CGM5Min
```


## b. run case observer for recnum
```shell

# change `--group_id`
# change `--rec_ckpd_list`: use CGM5Min-Bf24H amd CGM5Min-Af2H
# change `--test`: to false
python run_caseobs_recckpd_tkn.py \
    --group_id 0 \
    --scope_type whole \
    --rec_ckpd_list CGM5Min-Bf24H \
    --case_tkn RecNum \
    --test true
```

## c. run case filter for sftflt for whole

```shell
# change the `--group_id` to other number
python run_case_scope_filter.py \
    --group_id 1 \
    --scope_type whole \
    --recckpd_casetkn_observer_list CGM5Min-Bf24H-RecNum CGM5Min-Af2H-RecNum \
    --case_filter sftflt
```

## d run case observer `CGM5Min-N2CinTkn_tknidx`` on sftflt

unix shell version

```shell
# `--group_id`: change to other number
# `scope_type`: we use sftflt
# `rec_ckpd_list`: 'CGM5Min-Bf24H', can also be 'CGM5Min-Af2H'
# `sfx_list`: 'tknidx'; eg. CGM5Min-Bf24H_tknidx
# 'case_tkn': 1TknIn5Min
python run_caseobs_recckpd_tkn.py \
    --group_id 1 \
    --scope_type sftflt \
    --rec_ckpd_list 'CGM5Min-Bf24H' \
    --value_columns 'CGM5Min-N2Cin1Tkn' \
    --sfx_list 'tknidx' \
    --case_tkn '1TknIn5Min' \
    --batch_size 500 \
    --test false 
```

powershell version

```powershell
python .\run_caseobs_recckpd_tkn.py `
    --group_id 1 `
    --scope_type sftflt `
    --rec_ckpd_list CGM5Min-Af2H `
    --value_columns 'CGM5Min-N2Cin1Tkn' `
    --sfx_list 'tknidx' `
    --case_tkn '1TknIn5Min' `
    --batch_size 500 `
```

# 4. Prepare AI Dataset

```powershell
# You can ask GPT to change this to shell version.
python .\run_case_split_aidataset.py `
    --group_id 23 `
    --scope_type 'sftflt' `
    --task "CGMGPT" `
    --case_tkn_name_list 'BfCGM:CGM5Min*Bf24H*N2Cin1*tknidx' 'AfCGM:CGM5Min*Af2H*N2Cin1*tknidx' `
    --downsample_ratio 0.1 `
    --out_ratio 0.1 `
    --test_ratio 0.1 `
    --valid_ratio 0.1
```

# 5. Check Dataset

```shell
python run_clm_cgmgpt.py \
    --check_dataset_only true \
    --model_name_or_path gpt2 \
    --tokenizer_name "Model_CGMGPT/tokenizer/CGM5Min-N2Cin1Tkn.json" \
    --dataset_name 'CGMGPT-CGM5MinBf24HN2Cin1tknidx-CGM5MinAf2HN2Cin1tknidx-dsmp0.1-out0.1-test0.1-valid0.1' \
    --train_set_selector "in_train:C1&t2" \
    --eval_set_selectors "in_valid:C1&t1" "in_valid:C1&t2" "in_test:C1&t1" "in_test:C1&t2" "out_test:C1&t1"  "out_test:C1&t2" "out_whole:C1&t1"  "out_whole:C1&t2" \
    --output_dir "Model_CGMGPT/Model/C1&t2-bs64xga4x2gpus" \
    --overwrite_output_dir false \
    --num_train_epochs 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --logging_steps 1 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 5 \
    --report_to wandb \
    --preprocessing_num_workers 4 \
```




# 6. Run Model

## a. Precode

```shell
du -s -h --block-size=G Model_CGMGPT/AIDataSet/
tmux new -s jluo41
tmux attach -t jluo41
# partition: queue
# gpus: number of gpus
# mem: the size of mem
# cpus-per-task: as the name shows
# time: how long to set the srun, we might set it as 24:00:00
srun --partition=a100 --gpus=2 --mem=12GB --cpus-per-task=4 --time=8:00:00 --pty /bin/bash
srun --partition=a100 --gpus=2 --mem=24GB --cpus-per-task=4 --time=14:00:00 --pty /bin/bash
export WANDB_PROJECT="cgmgpt-v2"
cd workspace/WellDoc-CgmGPTv2-WorkSpace/
conda activate torch
```

## b. in_train:C1&t2

```shell
python run_clm_cgmgpt.py \
     --train_set_selector "in_train:C1&t2" \
    --output_dir "Model_CGMGPT/Model/C1&t2-bs64xga4x2gpus" \
    --check_dataset_only false \
    --model_name_or_path gpt2 \
    --tokenizer_name "Model_CGMGPT/tokenizer/CGM5Min-N2Cin1Tkn.json" \
    --dataset_name 'CGMGPT-CGM5MinBf24HN2Cin1tknidx-CGM5MinAf2HN2Cin1tknidx-dsmp0.1-out0.1-test0.1-valid0.1' \
    --eval_set_selectors "in_valid:C1&t1" "in_valid:C1&t2" "in_test:C1&t1" "in_test:C1&t2" "out_test:C1&t1"  "out_test:C1&t2" "out_whole:C1&t1"  "out_whole:C1&t2" \
    --overwrite_output_dir false \
    --num_train_epochs 8 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_eval_samples 1280 \
    --save_steps 200 \
    --save_total_limit 5 \
    --report_to wandb \
    --preprocessing_num_workers 4 \

```

## c. in_train:C1&tw1

```shell
python run_clm_cgmgpt.py \
    --train_set_selector "in_train:C1&t1" \
    --output_dir "Model_CGMGPT/Model/C1&t1-bs64xga4x2gpus" \
    --check_dataset_only false \
    --model_name_or_path gpt2 \
    --tokenizer_name "Model_CGMGPT/tokenizer/CGM5Min-N2Cin1Tkn.json" \
    --dataset_name 'CGMGPT-CGM5MinBf24HN2Cin1tknidx-CGM5MinAf2HN2Cin1tknidx-dsmp0.1-out0.1-test0.1-valid0.1' \
    --eval_set_selectors "in_valid:C1&t1" "in_valid:C1&t2" "in_test:C1&t1" "in_test:C1&t2" "out_test:C1&t1"  "out_test:C1&t2" "out_whole:C1&t1"  "out_whole:C1&t2" \
    --overwrite_output_dir false \
    --num_train_epochs 8 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_eval_samples 1280 \
    --save_steps 200 \
    --save_total_limit 5 \
    --report_to wandb \
    --preprocessing_num_workers 4 \
```

## d. in_train:C1

```shell
python run_clm_cgmgpt.py \
    --train_set_selector "in_train:C1" \
    --output_dir "Model_CGMGPT/Model/C1-bs64xga4x2gpus" \
    --check_dataset_only false \
    --model_name_or_path gpt2 \
    --tokenizer_name "Model_CGMGPT/tokenizer/CGM5Min-N2Cin1Tkn.json" \
    --dataset_name 'CGMGPT-CGM5MinBf24HN2Cin1tknidx-CGM5MinAf2HN2Cin1tknidx-dsmp0.1-out0.1-test0.1-valid0.1' \
    --eval_set_selectors "in_valid:C1&t1" "in_valid:C1&t2" "in_test:C1&t1" "in_test:C1&t2" "out_test:C1&t1"  "out_test:C1&t2" "out_whole:C1&t1"  "out_whole:C1&t2" \
    --overwrite_output_dir false \
    --num_train_epochs 8 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --logging_steps 1 \
    --eval_steps 100 \
    --max_eval_samples 1280 \
    --save_steps 200 \
    --save_total_limit 5 \
    --report_to wandb \
    --preprocessing_num_workers 4 \
```

