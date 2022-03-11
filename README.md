
*Environment setup
``` bash
conda env create -f env.yaml
```

*Preparing and note
Download "mathematics_dataset" from https://console.cloud.google.com/storage/browser/mathematics-dataset
The experiment is implemented using GPU with python3.8 and cuda>=11

change variables in math_ds_dgl_transformer.py:
1. Change "exp_name" and "unique_id"
2. Change "mdsmgr" to the file path of "mathematics_dataset"
3. Change "ds" and "ds_interpolate" to the selected module


*Usage (training and inference)
``` bash
python math_ds_dgl_transformer.py
```

For more information, please see the reference.md
