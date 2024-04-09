# TableDC

This repo provides the following:
The implementation of TableDC, a deep clustering algorithm for data cleaning and integration, specifically schema inference, entity resolution, and domain discovery.

This project comprises two major components, each addressing different data integration challenges: schema inference, entity resolution, and domain discovery:
* For schema inference, we use [SBERT](https://www.sbert.net/docs/hugging_face.html), [FastText](https://fasttext.cc/docs/en/crawl-vectors.html), USE, and [TabTransformer](https://github.com/jrzaurin/pytorch-widedeep) to embed tables.
* For entity resolution, we employ [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi) and [SBERT](https://www.sbert.net/docs/hugging_face.html) to embed rows.
* For domain discovery, we utilize T5 and [SBERT](https://www.sbert.net/docs/hugging_face.html) to embed columns.
* The generated dense embedding matrix `(X.txt)` will then serve as the input for clustering in the TableDC.

## Prerequisites
* Python 3.x
* PyTorch
* NumPy
* scikit-learn

## Dataset
The dataset comprises embeddings stored in `data/X.txt` and ground truth labels in `data/label.txt`.

- **Special thanks** to the authors (of baseline deep clustering and embedding approaches mentioned in the paper) for providing their implementations publicly available.

## Steps for Reproducing Results

This demo outlines steps to reproduce results for schema inference, entity resolution and domain discovery with TableDC. One ready-to-use vector X.txt (for schema Inference) is given in `data/`. Just use Step 3 below to get the clustering results.

### Schema Inference

1. **Schema-Level Web Tables Data (SBERT + TableDC):**
   - Process `schema inference/schema + instances/Preprocessing.ipynb` to extract schema-level information from tables.

2. **Generating Embedding Matrix:**
   - Use the generated `TextPre1.csv` to produce a dense embedding matrix (`X.txt`) with SBERT by running `schema inference/schema only/SBERT+FastText.py`.

3. **Clustering with TableDC:**
   - Utilize `X.txt` feature vector for clustering in TableDC:
     - Navigate to the `data/` and run the pretraining script `(pretrain_ae.py): python pretrain_ae.py --pretrain_path data/X.txt`.
     - Ensure that the pretrained model `(X.pkl)` and dataset `(X.txt and label.txt)` are in the `data/ directory`.
     - Run the script `(TableDC.py): python TableDC.py --pretrain_path data/X.pkl --name X`.
     - Update `nb_dimension = 768` accordingly (for SBERT use 768).

### Entity Resolution

1. **Row Embedding Matrix for Entity Resolution:**
   - Run `entity resolution/ER.py` to obtain row embedding matrix (`X.txt`) using EmbDi, and `entity resolution/ER_SBERT/ER_SBERT.py` for row embeddings with SBERT.

2. **Clustering for Entity Resolution:**
   - Repeat step (3) from Schema Inference with the row embedding matrix as input.

### Domain Discovery

- For domain discovery, refer to the respective folders (`schema inference/`, `entity resolution/`, and `domain discovery/`) or `full_data/` for a combination of all embeddings (except DD due to size limits). Ready-to-use embeddings are provided for FastText.
- Example: Compile `domain discovery/DD.py` for column embedding matrix (`X.txt`) using EmbDi, and `domain discovery/DD_SBERT(H+B)/DD_SBERT

## Standard Clustering Algorithms

- To get results with standard clustering algorithms, execute `SC/SC.py`.

## Note

1. **Data Preparation:**
   Due to storage limitations, please unzip `Tables.zip` before compiling the schema inference code.

2. **Computational Environment:**
   Due to different levels of precision in floating-point arithmetic and architectural aspects of different GPUs and CPUs, the resulting values can be slightly different. However, the overall results will remain consistent.

## Model Architecture and Training Configuration

### Autoencoder Architecture

The Autoencoder (AE) in TableDC contains four main layers, with each layer's size specified as follows:
- **Encoder Layer:**
  - Input Dimension: `n_input`
  - Output Dimension: `n_enc_1`
- **Latent Layer:**
  - Input Dimension from Encoder Layer: `n_enc_1`
  - Output Dimension (Latent Space): `n_z`
- **Decoder Layer:**
  - Input Dimension from Latent Layer: `n_z`
  - Output Dimension: `n_dec_1`
- **Output Layer:**
  - Input Dimension from Decoder Layer: `n_dec_1`
  - Output Dimension (Reconstructed Input): `n_input`

### Training Configuration

- **Optimization Algorithm:** Adam optimizer.
- **Learning Rate:** Configurable via `--lr`. Default set to `1e-3`.
- **Loss Function:** Combination of Kullback–Leibler divergence and Mean Squared Error loss.
- **Number of Epochs:** Please see Section 4.1.4 in the paper.
   
