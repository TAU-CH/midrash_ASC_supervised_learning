# ASC Dataset v2

## Overview

ASC_dataset_v2 is an updated version of the ASC dataset with improved balance, additional manuscripts, and a refined train/validation/test split structure. This version includes **284 images** (up from 238 in v1) with perfect balance across all splits.

## Key Changes from v1 (ASC_dataset)

### 1. Added Manuscripts

**MS 42 (French)** - 23 images
- **Resolution**: 8.23 MP (HIGHER than average - 130% of dataset average)
- **Quality**: NOT low-resolution
- **Annotations**: Automatic (test set annotations)
- **Status**: Added to TRAIN set

**MS 33 (German)** - 23 images (1 image excluded: `033_003_05`)
- **Resolution**: 5.24 MP (acceptable quality)
- **Quality**: NOT low-resolution
- **Annotations**: Manual (has character-level annotations)
- **Status**: Added to TRAIN set (1 image excluded to match MS 42 count)

### 2. Manuscript Movements

To maintain 80/20 train/test split and manuscript-level certification:

**Moved from TRAIN to TEST:**
- **MS 1** (French, 8 images) - moved to maintain balance
- **MS 6** (German, 8 images) - moved to maintain balance

### 3. Test Set Split into Val/Test

The original test set (58 images) has been split into:
- **Val**: 26 images (44.8% of original test)
- **Test**: 32 images (55.2% of original test)

This provides a validation set for hyperparameter tuning and model selection while maintaining a held-out test set.

## Dataset Statistics

### Overall Statistics

| Split | Script Subtype | Manuscripts | Total Images | Avg per MS | % of Split | % of Total |
|-------|----------------|-------------|--------------|------------|------------|------------|
| **TRAIN** | FRENCH | 13 | 113 | 8.69 | 50.00% | 39.79% |
| **TRAIN** | GERMAN | 13 | 113 | 8.69 | 50.00% | 39.79% |
| **VAL** | FRENCH | 2 | 13 | 6.50 | 50.00% | 4.58% |
| **VAL** | GERMAN | 2 | 13 | 6.50 | 50.00% | 4.58% |
| **TEST** | FRENCH | 5 | 16 | 3.20 | 50.00% | 5.63% |
| **TEST** | GERMAN | 4 | 16 | 4.00 | 50.00% | 5.63% |

**Summary:**
- **Total Images**: 284 (226 train + 26 val + 32 test)
- **Train Split**: 79.6% (226 images) - Perfectly balanced (113 French = 113 German)
- **Val Split**: 9.2% (26 images) - Perfectly balanced (13 French = 13 German)
- **Test Split**: 11.3% (32 images) - Perfectly balanced (16 French = 16 German)
- **Manuscripts**: 26 in train, 4 in val, 9 in test

## Split Logic and Rationale

### Train/Test Split (80/20)

The dataset maintains an **80/20 train/test split** at the manuscript level using the `balanced_80_20` split method:
- **Split Method**: Manually curated manuscript lists (`TRAIN_*_IDS` and `TEST_*_IDS`) to achieve approximately 80% train / 20% test ratio
- **Train Set**: High-resolution manuscripts selected to maximize dataset size while maintaining quality
  - **Annotation Types**: Includes both manually and automatically annotated manuscripts
    - Manual annotations: 119 images (e.g., MS 4, 7, 8, 12, 13, 16, 17, 20, 22, 24, 27, 28, 29, 33, 40, 46, 58)
    - Automatic annotations: 107 images (e.g., MS 11, 14, 18, 19, 31, 41, 42, 43, 45)
  - Examples: MS 42 (automatic annotations) and MS 33 (manual annotations) are both in train
- **Test Set**: Manuscripts selected to provide a held-out evaluation set
  - **Annotation Types**: Currently all automatically annotated manuscripts (32 images)
  - Examples: MS 1, 6, 9, 10, 15, 21, 23, 25, 26, 32, 44, 51, 52
- **Manuscript-level certification**: No manuscript ID appears in both train and test
- **Balance**: Equal number of French and German images in both train and test sets

**Note**: The split is **not** based solely on annotation type. The train set contains both manual and automatic annotations, while the test set currently contains only automatic annotations. The split is manually curated to achieve the 80/20 ratio and perfect balance.

### Test Set Split into Val/Test

The original test set (58 images) was split into validation and test sets using the following criteria:

#### Requirements:
1. **Perfect Balance**: Same number of French and German images in each split
2. **Manuscript-level Certification**: No manuscript ID appears in both val and test
3. **Equal Manuscript Counts**: As close as possible to equal number of manuscripts per script subtype
4. **50/50 Split**: As close as possible to 50% val, 50% test
5. **Minimal Image Loss**: All images from original test set are included

#### Optimal Split (Selected):

**Val Set** (26 images, 44.8% of original test):
- **French**: MS 1 (8 images) + MS 26 (5 images) = **13 images**
- **German**: MS 6 (8 images) + MS 23 (5 images) = **13 images**
- **Manuscripts**: 2 French, 2 German (equal counts)
- **Balance**: Perfect (0 difference)

**Test Set** (32 images, 55.2% of original test):
- **French**: MS 9 (4) + MS 10 (4) + MS 44 (4) + MS 51 (2) + MS 52 (2) = **16 images**
- **German**: MS 15 (4) + MS 21 (4) + MS 25 (4) + MS 32 (4) = **16 images**
- **Manuscripts**: 5 French, 4 German (1 difference - unavoidable due to discrete manuscript sizes)
- **Balance**: Perfect (0 difference)

**Note on Manuscript Count Imbalance**: The test set has 5 French manuscripts vs 4 German manuscripts (1 difference). This is unavoidable given the discrete manuscript sizes available (8, 5, 4, 4, 4, 2, 2 for French and 8, 5, 4, 4, 4, 4 for German). However, the **image counts are perfectly balanced** (16 French = 16 German), which is the primary requirement for model training and evaluation. The slight manuscript count imbalance does not affect the statistical validity of the test set.

#### Why This Split?

1. **Perfect Balance**: Both val and test have exactly equal French/German **images** (13/13 and 16/16)
2. **Manuscript Certification**: No overlap between val and test manuscript IDs
3. **Close to 50/50**: 26/32 split is only 3 images off from perfect 29/29 split
4. **No Image Loss**: All 58 original test images are included
5. **Equal MS Counts in Val**: Val has equal manuscript counts (2 French = 2 German)

**Note**: A perfect 29/29 split is mathematically impossible given the discrete manuscript sizes (8, 5, 4, 4, 4, 2, 2 for French and 8, 5, 4, 4, 4, 4 for German) while maintaining perfect balance and manuscript-level certification.

## Directory Structure

```
datasets/ASC_dataset_v2/
├── train/
│   ├── french/ (113 images)
│   └── german/ (113 images)
├── val/
│   ├── french/ (13 images)
│   └── german/ (13 images)
├── test/
│   ├── french/ (16 images)
│   └── german/ (16 images)
└── mappings/
    ├── train_mapping.json
    ├── val_mapping.json
    └── test_mapping.json
```

## Train Set Manuscripts

**French** (13 manuscripts, 113 images):
- MS 8, 11, 14, 16, 19, 20, 27, 29, 31, 33, 41, 42, 43, 45, 46

**German** (13 manuscripts, 113 images):
- MS 4, 7, 12, 13, 17, 18, 22, 24, 28, 33, 40, 58

## Validation Set Manuscripts

**French** (2 manuscripts, 13 images):
- MS 1: 8 images
- MS 26: 5 images

**German** (2 manuscripts, 13 images):
- MS 6: 8 images
- MS 23: 5 images

## Test Set Manuscripts

**French** (5 manuscripts, 16 images):
- MS 9: 4 images
- MS 10: 4 images
- MS 44: 4 images
- MS 51: 2 images
- MS 52: 2 images

**German** (4 manuscripts, 16 images):
- MS 15: 4 images
- MS 21: 4 images
- MS 25: 4 images
- MS 32: 4 images

**Note**: The test set has unequal manuscript counts (5 French vs 4 German) but equal image counts (16 French = 16 German). This manuscript count imbalance is unavoidable due to the discrete manuscript sizes available, but does not affect the statistical validity of the test set since image-level balance is maintained.

## Excluded Images

**MS 33**: Image `033_003_05` was excluded to balance MS 33 (24 images) with MS 42 (23 images), resulting in 23 images each.

## Image Normalization

**Yes, all images in ASC_dataset_v2 are normalized.**

All images have been processed through the normalization pipeline:
- **Source**: Images are copied from `data/normalized/` directory (normalized images)
- **Target Height**: Average character height normalized to **70 pixels**
- **Method**: Standard relative height filtering with Otsu binarization
- **Scale Factors**: Applied to achieve target character height (stored in metadata)

### Special Normalization Handling

**Image `029_000_01`** receives special treatment during normalization due to its unusual characteristics:

- **Issue**: This image has unusually large text relative to image height (max component height ~22% of document height, vs. typical ~5-10%)
- **Normal Parameters** (for all other images):
  - `relative_height_min`: 0.01 (1% of document height)
  - `relative_height_max`: 0.10 (10% of document height)
- **Special Parameters** (for `029_000_01` only):
  - `relative_height_min`: 0.02 (2% of document height)
  - `relative_height_max`: 0.30 (30% of document height)

**Rationale**: The normal maximum threshold (10%) would filter out too many valid character components in this image. The special parameters (30% max) allow larger components to be included in the height calculation, ensuring accurate normalization while maintaining quality control standards.

**Result**: Image `029_000_01` successfully normalizes to 72.03px average character height (within ±5px tolerance) with 66 connected components, passing all QC checks.

## Quality Control

All images in this dataset have passed quality control checks:
- **Normalization**: Average character height ≈ 70px (tolerance: ±5px)
- **Component Count**: Minimum 20 connected components per image
- **Scale Factor**: Reasonable (not extreme up/down scaling)
- **QC Status**: All images passed quality control (qc_passed = True)

## Usage

This dataset is designed for:
1. **Training**: Use `train/` split for model training
2. **Validation**: Use `val/` split for hyperparameter tuning and model selection
3. **Testing**: Use `test/` split for final evaluation (held-out, never used during development)

## Comparison with v1

| Metric | v1 (ASC_dataset) | v2 (ASC_dataset_v2) | Change |
|--------|------------------|---------------------|--------|
| **Total Images** | 238 | 284 | +46 (+19.3%) |
| **Train Images** | 196 | 226 | +30 (+15.3%) |
| **Test Images** | 42 | 58 (val+test) | +16 (+38.1%) |
| **Manuscripts** | 24 | 39 | +15 (+62.5%) |
| **Balance** | Perfect | Perfect | Maintained |
| **Split Ratio** | 82.4% / 17.6% | 79.6% / 9.2% / 11.3% | Improved |

## Metadata Files

- **Main metadata**: `data/main_region_dataset/splits_balanced_80_20/metadata_all_v2_train_val_test.csv`
- **Mapping files**: `datasets/ASC_dataset_v2/mappings/*_mapping.json`

## Notes

- This version maintains **manuscript-level certification** - no manuscript appears in more than one split
- All splits are **perfectly balanced** (equal French/German images)
- The test split into val/test was optimized to maximize balance while maintaining manuscript-level certification
- MS 42 and MS 33 were added because they are NOT low-resolution and provide valuable additional data

## Citation

If you use this dataset, please cite the original ASC dataset and note that this is version 2 with additional manuscripts and refined splits.

