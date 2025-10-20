# README – CIFAR-100 Vision Transformer Hybrid Classifier

## Overview
This project implements a **hybrid CNN + Transformer model** for image classification on the **CIFAR-100 dataset**.  
The network combines convolutional feature extraction with a Transformer encoder to learn spatial and contextual relationships.  
It includes data augmentation, Mixup, learning rate scheduling, visualization (UMAP, confusion matrix), and per-class accuracy reporting.

---

## Key Features
- **Hybrid Architecture:** Convolutional backbone + Transformer encoder.
- **Mixup & CutMix augmentation** via `timm.data.mixup`.
- **OneCycleLR scheduler** for adaptive learning rate.
- **Visualization tools:**  
  - Loss curve  
  - Confusion matrix (raw, normalized, and log-scaled)  
  - UMAP feature embeddings  
- **CIFAR-100 compatible:** Predefined mean and std normalization.
- **Batch-efficient:** Uses `pin_memory`, `persistent_workers`, and `prefetch_factor` for faster dataloading.

---

## Requirements
Install dependencies before running:

    pip install torch torchvision timm umap-learn scikit-learn matplotlib numpy pandas

Ensure CUDA is available for GPU acceleration:

    python -c "import torch; print(torch.cuda.is_available())"

---

## Model Architecture
`Net` consists of:
1. **Conv Layers:**  
   - Depthwise and pointwise convolutions with BatchNorm and GELU activations.  
   - Two downsampling steps.
2. **Transformer Encoder:**  
   - 7 layers, 16 heads, 256-d hidden dimension.  
   - Includes positional embeddings and a learnable class token.
3. **Head:**  
   - LayerNorm → Linear (256→512→100) with GELU + Dropout(0.2).  
   - Output dimension = 100 (CIFAR-100 classes).

---

## Training Configuration
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- **Scheduler:** OneCycleLR with cosine annealing
- **Loss Function:** SoftTargetCrossEntropy
- **Augmentations:**
  - RandomCrop, HorizontalFlip
  - RandAugment
  - ColorJitter
  - RandomErasing

---

## File Outputs
| File | Description |
|------|--------------|
| `loss.png` | Training loss curve |
| `confusion_matrix.png` | Raw confusion matrix |
| `confusion_matrix_normalized.png` | Normalized matrix |
| `umap_test_features.png` | 2D UMAP embedding |
| `CIFAR100v2(200)_result.pth` | Model weights after training |

---

## Functions Summary

| Function | Purpose |
|-----------|----------|
| `train()` | Runs model training with Mixup and loss recording |
| `model_SL(PATH, load)` | Saves or loads model weights |
| `loss_curves()` | Plots and saves training loss over epochs |
| `random_sample_show()` | Displays 6 random predictions |
| `test_acc()` | Returns test accuracy summary |
| `eval_preds()` | Collects true and predicted labels |
| `confusion_matrix()` | Builds a 100×100 confusion matrix |
| `plot_confusion_matrix()` | Visualizes confusion matrix variants |
| `class_accuracies()` | Reports per-class accuracy |
| `collect_features()` | Extracts feature embeddings |
| `run_umap_visualization()` | Performs UMAP dimensionality reduction |

---

## Usage
### 1. Train the model
    train_enable = True
    epochs = 200
The script will train for 200 epochs, log loss, and save results.

### 2. Load pretrained weights
    train_enable = False
    model_SL('./CIFAR100v2(200)_result.pth', load=True)

### 3. Generate visualizations
Automatically produces:
- Loss curve  
- Confusion matrices  
- UMAP projection  

---

## Example Output
'''Console
cuda
[001, 00100] loss: 3.996
[001, 00200] loss: 3.873
...
Finished Training
Overall accuracy from CM: 78.25%
Top 10 classes:
09 baby: 95.2%
...
Saved umap_test_features.png
'''
---

## Notes
- Training 200 epochs on CIFAR-100 requires ~4–6 hours on a modern GPU.  
- UMAP visualization may take several minutes.  
- Modify `batch_size` or `num_workers` for hardware constraints.

---

## License

# MIT License

Copyright (c) 2025 Yushing Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
