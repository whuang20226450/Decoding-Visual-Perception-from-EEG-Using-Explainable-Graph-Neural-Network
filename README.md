# Instructions

**Warning:** This code may be messy and could contain bugs. This code is just for reference.

## 1. Download the Data  
Download the dataset from [Stanford PURL](https://purl.stanford.edu/bq914sc3730) and extract the files. Ensure you have the `S1~10.mat` files available.

## 2. Preprocess the Data  
Navigate to the `data` directory and run the following script to generate `.npy` files:  

```bash
cd data
python preprocess_v2.py
```

This will create `S1~10.npy`.

## 3. Run Experiments  
Navigate to the `training` directory and execute the experiment scripts:  

```bash
cd training
python exp1.py
python exp2.py
```

These scripts reproduce the experiments described in the paper.

## 4. Explainability Analysis  
Navigate to the `explainer` directory and run the following scripts in order:  

1. **Generate XAI Input Data**  

    ```bash
    cd explainer
    python genXaiInput_all.py
    ```

2. **Run GNN Explainer**  

    ```bash
    python GNNexplainer_all.py
    ```

3. **Plot Explainability Results**  

    ```bash
    python plotXai.py
    ```

This process will generate and visualize explainability results.
