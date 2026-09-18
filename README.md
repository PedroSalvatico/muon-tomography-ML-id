# muon-tomography-ML-id

An experimental tomography and material-classification pipeline based on muon
scattering simulations. It combines Point of Closest Approach (PoCA)
reconstruction, voxel-level feature extraction, and machine-learning models to
classify aluminum, copper, and tungsten.

## Pipeline

```text
raw simulation logs (.log)
          |
          v
datav4.py + log_file_list.txt
  - extracts detector tracks
  - calculates scattering angles and PoCA points
  - aggregates PoCA density, mean angle, and standard deviation per voxel
  - generates four rotations of each simulation
          |
          v
AI_X_Sample_0.pkl
shape: (792, 2, 2, 2, 3)
          |
          +--> k_neigh_v2.py
          +--> neural_network_v2.py
          +--> cnnV1.py
```

The processed dataset contains 198 simulations with four orientations each,
for a total of 792 samples. Every sample is represented as a `2 x 2 x 2` voxel
grid with three features per voxel:

1. relative PoCA density;
2. mean scattering angle;
3. scattering-angle standard deviation.

The target labels are reconstructed from the eight digits encoded in each
filename listed in `log_file_list.txt`:

- `1`: aluminum;
- `2`: copper;
- `3`: tungsten.

## Files

| File | Purpose |
|---|---|
| `datav4.py` | Final feature-extraction pipeline. It reads the raw logs listed in `log_file_list.txt`, calculates the voxel features, creates rotated samples, and serializes the resulting tensor. |
| `AI_X_Sample_0.pkl` | Preserved final dataset containing 792 processed samples. |
| `log_file_list.txt` | Ordered manifest of the 198 simulations. The filenames also encode the material labels. |
| `k_neigh_v2.py` | K-nearest neighbors classifier with confusion matrix and ROC curves. |
| `neural_network_v2.py` | Dense neural-network classifier applied independently to the voxel features. |
| `cnnV1.py` | 3D convolutional neural network that classifies all eight voxels jointly. |

## Environment

The project was validated on September 17, 2026, in the Conda environment
`ml-env`, using Python 3.11.16 and CPU execution. The tested package versions
were:

- TensorFlow 2.21.0
- Keras 3.15.1
- NumPy 2.4.6
- SciPy 1.17.1
- scikit-learn 1.9.0
- Matplotlib 3.11.1
- seaborn 0.13.2
- pandas 3.0.5
- hilbertcurve 2.0.5

Create an equivalent environment with:

```bash
conda create -n ml-env python=3.11 -y
conda activate ml-env
python -m pip install \
  tensorflow==2.21.0 numpy==2.4.6 scipy==1.17.1 \
  scikit-learn==1.9.0 matplotlib==3.11.1 seaborn==0.13.2 \
  pandas==3.0.5 hilbertcurve==2.0.5
```

TensorFlow may report that CUDA drivers are unavailable when no compatible GPU
is configured. All three models also run on CPU.

## Running the models

Run the commands from the repository root. The models read
`AI_X_Sample_0.pkl` and `log_file_list.txt` directly, so the raw simulation logs
are not required for training and evaluation.

```bash
conda activate ml-env
python k_neigh_v2.py
python neural_network_v2.py
python cnnV1.py
```

On a machine without a graphical interface, set a non-interactive Matplotlib
backend, for example:

```bash
MPLBACKEND=Agg python k_neigh_v2.py
```

All three scripts completed successfully during validation. As a reference,
the KNN classifier achieved approximately 97.1% accuracy with the train/test
split defined in the script. Neural-network results may vary because their
weights are randomly initialized.

## Dataset availability

The raw `.log` files are the source data used by `datav4.py`. The local archive
does not contain the complete raw dataset: `log_file_list.txt` has 198 entries,
but only 4 of those logs remain available locally. Consequently, the full
processed dataset cannot currently be regenerated from the surviving logs.

This does not prevent the models from running because `AI_X_Sample_0.pkl` is a
preserved output of the complete extraction process. It already contains the
features produced from all 198 simulations.

When the complete raw dataset is available, the intended extraction command is:

```bash
python datav4.py
```

The script writes its result to `test.pkl`, while the final model scripts expect
the complete artifact to be named `AI_X_Sample_0.pkl`. This naming difference
is documented here without modifying the original source code.

## Repository contents

The final repository should contain exactly these seven files:

```text
README.md
datav4.py
AI_X_Sample_0.pkl
log_file_list.txt
k_neigh_v2.py
neural_network_v2.py
cnnV1.py
```

Intermediate implementations, exploratory scripts, notebooks, simulation
configuration files, `test.pkl`, caches, and raw `.log` files are intentionally
excluded. The local raw-data archive is incomplete, occupies approximately
9.3 GB, and contains 33 individual files larger than 100 MB.

A suitable `.gitignore` is:

```gitignore
__pycache__/
*.py[cod]
*.log
.ipynb_checkpoints/
```

If the complete raw dataset is recovered, it should be published separately in
a research-data repository such as Zenodo, OSF, or institutional storage, with
its DOI or URL referenced here. The compact `AI_X_Sample_0.pkl` artifact is
sufficient to execute the machine-learning pipeline.

## Reproducibility notes

- The scripts do not set every TensorFlow random seed, so neural-network metrics
  may differ between runs.
- The plots request the Arial font. Matplotlib falls back to another font when
  Arial is unavailable; this does not affect model execution.
- `k_neigh_v2.py`, `neural_network_v2.py`, and `cnnV1.py` were executed
  successfully without changing their original source code.
- The repository contains a Python pickle file. Only load pickle files obtained
  from a trusted source.
