
1. Estimate the signal and noise rato (SNR):

```bash
python tools/dataset_preparing/run_statistic.py \
--lotsa_path LOTSA/PATH
```

2. Export the datasets containing data snr > 20 via `snr.ipynb`.

3. Categorize the datasets into `selected_datasets.json`, and use `tools/dataset_preparing/build_dataset.py` to organize the datasets.

4. Generate the training and validation datasets by `spliting.py`.



