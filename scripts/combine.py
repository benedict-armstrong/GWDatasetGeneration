"""Combine raw GW HDF5 files into a single grouped file.

Usage:
    python -m ssm_bench.dataset.scripts.combine_gw_h5 \
        --input-dir data/bns/train_pv2 \
        --output data/bns/train_pv2/classification_combined_v2.h5 \
        --include-background
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if isinstance(val, h5py.Group):
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                try:
                    print(pre + "└── " + key + f" {val.shape}")
                except TypeError:
                    print(pre + "└── " + key + " (scalar)")
        else:
            if isinstance(val, h5py.Group):
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                try:
                    print(pre + "├── " + key + f" {val.shape}")
                except TypeError:
                    print(pre + "├── " + key + " (scalar)")


def combine_h5_files_grouped(
    true_files,
    false_files,
    output_path,
    channels: tuple[str, ...] = ("H1", "L1"),
    dataset_key="data",
    normalize_data=False,
    max_samples: int = -1,
) -> tuple[int, ...]:
    """Combines multiple HDF5 files into a single grouped HDF5 file.

    Output structure::

        foreground/
            waveforms/{channel}  (N, T)  per channel
            parameters/{key}     (N, ...)
        train_indices, val_indices  (written separately by the dataloader)

    Args:
        true_files: HDF5 files containing positive (signal) samples.
        false_files: HDF5 files containing negative (background) samples.
        output_path: Output HDF5 file path.
        channels: Channel names matching the channel axis of the source data.
        dataset_key: Key inside source HDF5 files storing the waveform array.
        normalize_data: Whether to z-score normalize waveform data.
        max_samples: Cap on total samples (-1 = use all).

    Returns:
        Shape of the combined dataset as ``(total_samples, num_channels, T)``.
    """
    all_files = [(p, 1) for p in true_files] + [(p, 0) for p in false_files]

    if len(all_files) == 0:
        raise ValueError("No files found")

    total_samples = 0
    param_keys: set[str] = set()
    for f, _ in all_files:
        with h5py.File(f, "r") as hf:
            total_samples += hf[dataset_key].shape[0]
            time_steps = hf[dataset_key].shape[-1]
            param_keys |= {k for k in hf.keys() if k not in [dataset_key, "raw_signal"]}

    if max_samples > 0:
        total_samples = min(total_samples, max_samples)

    num_channels = len(channels)

    # Normalization statistics
    data_mean, data_std = 0.0, 1.0
    if normalize_data:
        logger.info("Computing normalization statistics...")
        running_sum = 0.0
        running_sq_sum = 0.0
        n_elements = 0
        for fpath, _ in tqdm(all_files, desc="Computing stats"):
            with h5py.File(fpath, "r") as hf:
                data = hf[dataset_key][:]
                running_sum += np.sum(data)
                running_sq_sum += np.sum(data**2)
                n_elements += data.size
        data_mean = running_sum / n_elements
        data_std = np.sqrt(running_sq_sum / n_elements - data_mean**2)
        if data_std < 1e-8:
            data_std = 1.0
        logger.info(f"Normalization: mean={data_mean:.6f}, std={data_std:.6f}")

    # chunk_n = min(350, total_samples)

    with h5py.File(output_path, "w") as out_hf:
        wf_grp = out_hf.create_group("waveforms")
        pm_grp = out_hf.create_group("parameters")

        wf_dsets = {}
        for ch in channels:
            wf_dsets[ch] = wf_grp.create_dataset(
                ch,
                shape=(total_samples, time_steps),
                dtype=np.float32,
                chunks=(1, time_steps),
            )

        pm_dsets = {}
        chunk_n = 1
        for k in param_keys:
            pm_dsets[k] = pm_grp.create_dataset(
                k, shape=(total_samples,), dtype=np.float32, chunks=(chunk_n,)
            )
        label_dset = pm_grp.create_dataset(
            "labels", shape=(total_samples,), dtype=np.int8, chunks=(chunk_n,)
        )

        if normalize_data:
            out_hf.attrs["data_mean"] = data_mean
            out_hf.attrs["data_std"] = data_std
            out_hf.attrs["normalized"] = True
        else:
            out_hf.attrs["normalized"] = False

        idx = 0
        for fpath, label in tqdm(all_files, desc="Combining files"):
            if idx >= total_samples:
                break
            with h5py.File(fpath, "r") as hf:
                data = hf[dataset_key][:]  # (n, C, T)
                if normalize_data:
                    data = (data - data_mean) / data_std

                n = data.shape[0]
                n = min(n, total_samples - idx)

                for ci, ch in enumerate(channels):
                    wf_dsets[ch][idx : idx + n] = data[:n, ci]

                for k in param_keys:
                    if k in hf:
                        pm_dsets[k][idx : idx + n] = hf[k][:n]
                    else:
                        pm_dsets[k][idx : idx + n] = np.nan * np.ones(n)

                label_dset[idx : idx + n] = label
                idx += n

    logger.info(f"Combined grouped HDF5 saved to {output_path}")
    return (total_samples, num_channels, time_steps)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Combine raw GW HDF5 files into a single grouped file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw .h5 files (sig_*, bkg_*)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        default="combined.h5",
        help="Output path for the combined HDF5 file",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["H1", "L1"],
        help="Channel names (default: H1 L1)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Z-score normalize waveform data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Cap on total samples (-1 = use all)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    h5_files = list(input_dir.glob("*.h5"))

    signal_files = sorted(
        [fn for fn in h5_files if "sig" in fn.name],
        key=lambda x: int(x.stem.split("_")[1]),
    )

    background_files = sorted(
        [fn for fn in h5_files if "bkg" in fn.name],
        key=lambda x: int(x.stem.split("_")[1]),
    )

    logger.info(
        f"Found {len(signal_files)} signal files, {len(background_files)} background files"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    shape = combine_h5_files_grouped(
        signal_files,
        background_files,
        args.output,
        channels=tuple(args.channels),
        normalize_data=args.normalize,
        max_samples=args.max_samples,
    )

    logger.info(f"Combined dataset shape: {shape}")

    with h5py.File(args.output, "r") as hf:
        h5_tree(hf)


if __name__ == "__main__":
    main()
