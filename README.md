# Enhanced Pan-Tompkins Algorithm  

This repository is part of the [HIE grading with HRV signals](https://github.com/syu-kylin/HRVConformer) project and provides an **enhanced version of the Pan-Tompkins algorithm**. The enhanced algorithm improves HRV signal extraction from ECG signals, ensuring higher quality and availability, even in noisy and artifact-prone conditions. More details can be seen in our paper: [paper]().

```bibtex
@article{your_paper_reference,
  author    = {Your Name and Others},
  title     = {Enhanced Pan-Tompkins Algorithm for HRV Extraction from Noisy ECG Signals},
  journal   = {Journal Name},
  year      = {2025}
}
```

## Improvements Over the Standard Pan-Tompkins Algorithm  

A quantified comparison between the standard and enhanced versions of the Pan-Tompkins algorithm is shown in the table below. Using the **ANSeR dataset**, which includes **216 hours** (ANSeR1) and **259 hours** (ANSeR2) of ECG recordings, we evaluated HRV data availability after extraction, noise removal, and epoch generation.  

The **standard Pan-Tompkins algorithm** produced only **126 hours (5756 epochs)** and **161 hours (7207 epochs)** of usable HRV data. In contrast, our **enhanced Pan-Tompkins algorithm** preserved nearly all the original data, discarding only **1 hour** and **8 hours** of recordings, demonstrating its effectiveness in maintaining HRV data availability. 

### HRV Data Availability Comparison  

| Algorithm Version     | ANSeR1 (216h)        | ANSeR2 (259h)        |
|----------------------|--------------------|--------------------|
| **Total ECG Data**   | 216h               | 259h               |
| **Standard Pan-Tompkins**  | 126h (5756 epochs)   | 161h (7207 epochs)   |
| **Enhanced Pan-Tompkins**  | 215h (11,208 epochs) | 251h (13,067 epochs) |

---

### Examples of Extracted RR Intervals  

The following figures demonstrate some extracted RR intervals using enhanced algorithm and standard Pan-Tompkin:  
#### 1. noisy ECG processing
 ##### Example 1
 ![alt text](<figures/ID032 ANSeR1 36 HR rr interval comparison.svg>)

 ##### Example 2
 ![alt text](<figures/ID576 ANSeR1 06 HR rr interval comparison.svg>)

 ##### Example 3
 ![alt text](<figures/ID004 ANSeR2 36 HR rr interval comparison.svg>)

 ---

#### 2. long intervals processing
 ##### Example 1
 ![alt text](<figures/ID004 ANSeR2 12 HR rr interval comparison.svg>)

 ##### Example 2
 ![alt text](<figures/ID032 ANSeR1 24 HR rr interval comparison.svg>)

 ##### Example 3
 ![alt text](<figures/ID142 ANSeR1 12 HR rr interval comparison.svg>)

---

#### 3. standard Pan-Tompkin failed

 ##### Example 1
 ![alt text](<figures/ID751 ANSeR1 24 HR rr interval comparison.svg>)

 ##### Example 2
 ![alt text](<figures/ID558 ANSeR1 06 HR rr interval comparison.svg>)

---

More examples can be seen in [figures](./figures/).

## Installation and Usage  

To use this enhanced Pan-Tompkins algorithm, follow these steps:  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/syu-kylin/enhanced-Pan-Tompkin.git
   cd enhanced-Pan-Tompkin
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the algorithm on sample ECG data:**
   ```
   python rr_extraction.py --edf_fn "your edf file path and name" --chann_name "selected ECG channel"
   ```
   - **edf_fn**: (str) the edf file path including file name.
   - **chann_name**: (str or list) The selected could be a string for single channel or list of two string like ["ECG1", "ECG2"], ["X1", "X2"] for bipolar monatge. 
---

## Cite
If you find this work useful, please cite our paper:

```bibtex
@article{your_paper_reference,
  author    = {Your Name and Others},
  title     = {Enhanced Pan-Tompkins Algorithm for HRV Extraction from Noisy ECG Signals},
  journal   = {Journal Name},
  year      = {2025}
}
```

