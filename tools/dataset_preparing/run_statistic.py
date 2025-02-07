import os
import datasets
import logging
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from multiprocessing import Pool
import os
from tools.dataset_preparing.characteristics import (
    trend_and_seasonal_length,
    stationarity,
    shifting,
    transition,
    missing_rate,
    correlation,
    signal_noise_ratio,
)

from datasets import disable_caching
disable_caching()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"run_{current_time}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def split_dataset(dataset, n_splits):
    split_size = len(dataset) // n_splits
    split_datasets = []
    current_dataset = dataset

    if split_size != 0:
        for i in range(n_splits - 1):
            split = current_dataset.train_test_split(test_size=split_size)
            split_datasets.append(split["test"])
            current_dataset = split["train"]

    split_datasets.append(current_dataset)
    return split_datasets


STATISTIC2FUNC = {
    "mean": lambda x: np.nanmean(x),
    "std": lambda x: np.nanstd(x),
    "length": lambda x: len(x),
    "missing_rate": missing_rate,
    "snr": signal_noise_ratio,
    "shifting": shifting,
    "trend_and_seasonal_length": trend_and_seasonal_length,
    "stationarity": stationarity,
    "transition": transition,
    "correlation": correlation,
}


def statistic_analysis_per_series(series: dict, statitics: list):
    freq = series["freq"]
    freq = 32 if freq == "" else freq
    target = np.array(series["target"])
    assert target.ndim == 1

    series_meta_info = defaultdict(list)

    for statistic in statitics:
        tmp = None
        if series_meta_info == "trend_and_seasonal_length":
            tmp = STATISTIC2FUNC[statistic](target, freq)
        else:
            tmp = STATISTIC2FUNC[statistic](target)
        series_meta_info[statistic].append(tmp)

    return series_meta_info


def statistic_analysis_per_dataset(dataset: datasets.Dataset, statistics: list):
    assert "freq" in dataset.features
    assert "target" in dataset.features

    n_dim = dataset[0]["target"].ndim
    assert n_dim == 1 or n_dim == 2

    sample_num = len(dataset)
    n_var = dataset[0]["target"].shape[0] if n_dim == 2 else 1

    meta_info = defaultdict(list)
    meta_info["sample_num"] = [sample_num]
    meta_info["variance_num"] = n_var
    meta_info["freq"] = dataset[0]["freq"]
    
    # random sample some series to analyze
    random_statistic_num = round(0.1 * sample_num) if sample_num >= 1000 else 100
    random_statistic_num = min(sample_num, random_statistic_num)
    random_statistic_idxs = np.random.choice(
        sample_num, random_statistic_num, replace=False
    )

    for idx, sample in tqdm(enumerate(dataset)):
        if idx not in random_statistic_idxs:
            continue

        n_var = sample["target"].shape[0] if n_dim == 2 else 1
        sample_meta_info = dict()
        for i in range(n_var):
            target = sample["target"][i] if n_dim == 2 else sample["target"]
            series = {"freq": sample["freq"], "target": target}
            sample_meta_info[f"var{i}"] = statistic_analysis_per_series(
                series, statistics
            )

        for var_idx, series_info in sample_meta_info.items():
            if var_idx not in meta_info:
                meta_info[var_idx] = defaultdict(list)
            for statistic, statistic_value in series_info.items():
                meta_info[var_idx][statistic].extend(statistic_value)

    return meta_info


def statistic_analysis_per_dataset_parallel(
    dataset: datasets.Dataset, statistics: list, num_processes=8
):
    dataset_subsets = split_dataset(dataset, num_processes)

    with Pool(num_processes) as pool:
        results = pool.starmap(
            statistic_analysis_per_dataset,
            [(dataset_subset, statistics) for dataset_subset in dataset_subsets],
        )

    meta_info = defaultdict(list)

    for result in results:
        for key, value in result.items():
            if key in ("freq", "variance_num"):
                meta_info[key] = value
            elif key == "sample_num":
                meta_info[key].extend(value)
            elif "var" in key:
                if key not in meta_info:
                    meta_info[key] = defaultdict(list)
                for statistic, statistic_value in value.items():
                    meta_info[key][statistic].extend(statistic_value)

    for key, value in meta_info.items():
        if key in ("freq", "variance_num"):
            continue
        elif key == "sample_num":
            meta_info[key] = int(np.sum(value))
        elif "var" in key:
            for statistic, statistic_value in value.items():
                meta_info[key][statistic] = np.nanmean(statistic_value)

    return meta_info


def default(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lotsa_path", type=str)
    parser.add_argument("--dest_dir", type=str, default="./meta_info")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--statistics", type=str, nargs="+", default=["snr"])
    # 'mean', 'std', 'length', 'missing_rate', 'snr', 'shifting', 'trend_and_seasonal_length', 'stationarity', 'transition'
    # 'timepoint', 'min_length', 'max_length', 'mean_length'
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    LOTSA_PATH = args.lotsa_path
    DEST_DIR = args.dest_dir
    NUM_PROCESSES = args.num_processes
    STATISTICS = args.statistics

    if os.path.exists(DEST_DIR):
        print("Destination directory already exists! Whether to overwrite?")
        import pdb

        pdb.set_trace()
    else:
        print("Destination directory does not exist! Create it!")
        os.makedirs(DEST_DIR)

    if args.datasets is None:
        items = os.listdir(LOTSA_PATH)
        is_dir = lambda x: os.path.isdir(os.path.join(LOTSA_PATH, x))
        dataset_names = filter(is_dir, items)
        dataset_names = sorted(dataset_names)
    else:
        dataset_names = args.datasets

    if args.exclude is not None and os.path.exists(args.exclude):
        with open(args.exclude, "r") as f:
            exclude_list = f.read().splitlines()
        dataset_names = [name for name in dataset_names if name not in exclude_list]

    length = len(dataset_names)
    failed_list = []
    for i, dataset_name in enumerate(dataset_names):
        logger.info(f"{i+1}/{length}: {dataset_name} started!")
        try:
            dataset_path = os.path.join(LOTSA_PATH, dataset_name)
            dataset = datasets.load_from_disk(dataset_path).with_format("numpy")
            meta_info = statistic_analysis_per_dataset_parallel(
                dataset, STATISTICS, NUM_PROCESSES
            )
            dest_path = os.path.join(DEST_DIR, f"{dataset_name}.json")
            json.dump(meta_info, open(dest_path, "w"), default=default, indent=4)
        except Exception as e:
            failed_list.append(dataset_name)
            logger.warning(f"{dataset_name} failed!")
            logger.warning(e, exc_info=True)
            continue
        logger.info("finished!")
    logger.info(f"Failed list: {failed_list}")
