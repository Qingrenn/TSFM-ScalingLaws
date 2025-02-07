import os
import shutil
import datasets
import pandas as pd
from datasets import Features, Sequence, Value
from argparse import ArgumentParser

datasets.disable_caching()

LOTSA_PATH = ""
UTSD_PATH = ""
OUTPUT_PATH = ""


class DownSampleDataset:
    root = LOTSA_PATH
    output = OUTPUT_PATH
    dataset_dict = {"buildings_900k": 538577}

    def build(self, *args, **kwds):
        failed_list = []
        for dataset_name, downsample_ratio in self.dataset_dict.items():
            try:
                dataset = datasets.load_from_disk(self.root + "/" + dataset_name)
                new_dataset = dataset.train_test_split(
                    test_size=downsample_ratio, seed=0
                )["test"]
                new_dataset.save_to_disk(self.output + "/" + dataset_name, num_proc=8)
            except Exception as e:
                print(f"Error: {dataset_name} meets error: {e}")
                failed_list.append(dataset_name)
        return failed_list


class DownVarianceDataset:
    root = LOTSA_PATH
    output = OUTPUT_PATH
    dataset_dict = {
        "cmip6_1850": [
            11, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38
        ],
        "cmip6_1855": [
            11, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38
        ],
        "era5_1989": [0, 3, 4, 5, 6, 7, 8, 9, 17, 24, 25, 26, 27, 28, 29, 30],
        "era5_1990": [0, 3, 4, 5, 6, 7, 8, 9, 17, 24, 25, 26, 27, 28, 29, 30],
        "borg_cluster_data_2011": [1],
        "alibaba_cluster_trace_2018": [1],
        "PEMS04": [2],
        "PEMS08": [0, 2],
    }

    def build(self):
        for dataset_name, remained_dims in self.dataset_dict.items():
            print(dataset_name)
            failed_list = []
            try:
                dataset = datasets.load_from_disk(self.root + "/" + dataset_name)
                new_dataset = self._multi2uni_(dataset, remained_dims)
                new_dataset.save_to_disk(self.output + "/" + dataset_name)
            except Exception as e:
                print(f"Error: {dataset_name} meets error: {e}")
                failed_list.append(dataset_name)
        return failed_list

    def _multi2uni_(self, hf_dataset, remained_dims: list = None):

        def gen_uni_from_multi(dataset, remained_dims: list = None):
            for item in dataset:
                n_var = len(item["target"])
                for i in range(n_var):
                    if remained_dims is not None and i not in remained_dims:
                        continue
                    out_item = {
                        "item_id": item["item_id"] + "_dim" + str(i),
                        "start": item["start"],
                        "target": item["target"][i],
                        "freq": item["freq"],
                    }
                    yield out_item

        new_dataset = datasets.Dataset.from_generator(
            gen_uni_from_multi,
            gen_kwargs=dict(dataset=hf_dataset, remained_dims=remained_dims),
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                ),
            ),
            num_proc=8,
        )

        return new_dataset


class Utsd2LotsaDataset:
    root = UTSD_PATH
    output = OUTPUT_PATH
    dataset_dict = {
        "TDBrain": "0.002S",
        "MotorImagery": "0.001S",
        "IEEEPPG": "0.008S",
        "SelfRegulationSCP1": "0.004S",
        "SelfRegulationSCP2": "0.004S",
        "PigArtPressure": "S",
        "PigCVP": "S",
        "BIDMC32HR": "S",
        "AtrialFibrillation": "0.008S",
    }

    def build(self):
        utsd = datasets.load_from_disk(self.root)
        failed_list = []
        for dataset_name, freq in self.dataset_dict.items():
            try:
                dataset = utsd.filter(
                    lambda example: example["item_id"].startswith(
                        "Health_" + dataset_name
                    ),
                    num_proc=8,
                )
                new_dataset = self._utsd2lotsa_(dataset, freq)
                new_dataset.save_to_disk(self.output + "/" + dataset_name)
            except Exception as e:
                print(f"Error: {dataset_name} meets error: {e}")
                failed_list.append(dataset_name)
        return failed_list

    def _utsd2lotsa_(self, hf_dataset, freq):

        def gen_lotsa_from_utsd(dataset, freq):
            for item in dataset:
                yield {
                    "item_id": item["item_id"],
                    "start": (
                        item["start"] if item["start"] != "" else "2024-01-01 00:00:00"
                    ),
                    "target": item["target"],
                    "freq": item["freq"] if item["freq"] != "" else freq,
                }

        new_dataset = datasets.Dataset.from_generator(
            gen_lotsa_from_utsd,
            gen_kwargs=dict(dataset=hf_dataset, freq=freq),
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                ),
            ),
            num_proc=8,
        )

        return new_dataset


class CopyDataset:
    root = LOTSA_PATH
    output = OUTPUT_PATH
    dataset_list = [
        "LOOP_SEATTLE",
        "PEMS07",
        "PEMS_BAY",
        "Q-TRAFFIC",
        "bdg-2_bear",
        "bdg-2_fox",
        "bdg-2_panther",
        "covid19_energy",
        "elecdemand",
        "elf",
        "gfc14_load",
        "gfc17_load",
        "largest_2017",
        "largest_2018",
        "largest_2019",
        "largest_2020",
        "largest_2021",
        "pdb",
        "sceaux",
        "solar_power",
        "spain",
        "traffic_weekly",
        "kaggle_web_traffic_weekly",
        "azure_vm_traces_2017",
        "wiki-rolling_nips",
        "favorita_sales",
        "australian_electricity_demand"
    ]

    def build(self, soft=False):
        failed_list = []
        for dataset_name in self.dataset_list:
            try:
                if soft:
                    os.symlink(
                        self.root + "/" + dataset_name, self.output + "/" + dataset_name
                    )
                else:
                    shutil.copytree(
                        self.root + "/" + dataset_name, self.output + "/" + dataset_name
                    )
            except Exception as e:
                print(f"Error: {dataset_name} meets error: {e}")
                failed_list.append(dataset_name)
        return failed_list


LSF_PATH = "/data/qingrenyao/all_datasets/long_term_forecast/"
class LSFDataset:
    root = LSF_PATH
    output = '/data/qingrenyao/TSFMScalingLaws/dataset_test/LSF'
    dataset_list = [
        'ETT-small/ETTh1.csv',
        'ETT-small/ETTh2.csv',
        'ETT-small/ETTm1.csv',
        'ETT-small/ETTm2.csv',
        'weather/weather.csv',
        'electricity/electricity.csv',
    ]
    
    def build(self):
        failed_list = []
        for dataset_name in self.dataset_list:
            try:
                table = pd.read_csv(self.root + "/"+ dataset_name)
                new_dataset = self._lsf2lotsa_(table)
                new_name = dataset_name.split('/')[-1].split('.')[0]
                new_dataset.save_to_disk(self.output + "/" + new_name)
            except Exception as e:
                print(f"Error: {dataset_name} meets error: {e}")
                failed_list.append(dataset_name)
        return failed_list
    
    def _lsf2lotsa_(self, table:pd.DataFrame):
        
        def gen_lotsa_from_lsf(table):
            freq = pd.infer_freq(table['date'])
            if freq is None:
                freq = '10T'
            for col in table.columns[1:]:
                yield {
                    "item_id": col,
                    "start": table['date'][0],
                    "target": table[col].to_list(),
                    "freq": freq,
                }
        
        new_dataset = datasets.Dataset.from_generator(
            gen_lotsa_from_lsf,
            gen_kwargs=dict(table=table),
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                ),
            ),
        )
        
        return new_dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--build",
        nargs="+",
        default=["downsample", "downvariance", "utsd2lotsa", "copy"],
    )
    args = parser.parse_args()

    for b in args.build:
        if b == "downsample":
            obj = DownSampleDataset()
        elif b == "downvariance":
            obj = DownVarianceDataset()
        elif b == "utsd2lotsa":
            obj = Utsd2LotsaDataset()
        elif b == "copy":
            obj = CopyDataset()
        elif b == "lsf2lotsa":
            obj = LSFDataset()
        else:
            raise NotImplementedError

        print(f"Start build {b} datasets")
        obj.build()
