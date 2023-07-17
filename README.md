# marugoto_mods
My modifications to marugoto by Kather Lab (https://github.com/KatherLab/marugoto.git)

Also includes a version of highres-WSI-heatmaps (https://github.com/KatherLab/highres-WSI-heatmaps.git) that uses canny edge detection to remove artefacts.

Pre-processing script now includes an artefact detector to remove some tiles that get past the canny edge detector, modified from https://github.com/KatherLab/preprocessing-ng/tree/gecco_cohert.

Please reference their work if using this repo!

NOTE: this is a work in progress - additional functionality may be added as I discover things I need!

## Usage
Different scripts in this repo are used in different ways (for legacy reasons).

### mil_mods, vis_mods
These two are designed to be imported into another python script. For example:

```Python
from marugoto_mods.mil_mods import train_categorical_model_

train_categorical_model_mod(
        clini_table = clini,
        slide_csv = slide,
        feature_dir = feat_dir,
        target_label = target,
        output_path = outpath,
        lr = lr
    )
```

Further information can be found in the original repo: https://github.com/KatherLab/marugoto.git

The other files should be run from the command line as follows.
### pre-processing
1. Move to the marugoto_mods folder.
2. Activate an appropriate venv.
3. Run the following command:

```
python -m marugoto_mods.pre-processing \
    --outdir PATH/TO/SAVE/TILES/TO \
    --tile_size INT \
    --um_per_tile FLOAT \
    --brightness_cutoff INT \
    PATH/TO/WSIs
```
  Defaults are:
  - tile_size = 224
  - um_per_tile = 256.0
  - brightness_cutoff = 224

### ViT_extractor
1. Move to the marugoto_mods folder.
2. Activate an appropriate venv.
3. Run the following command (augmented_repetitions is optional):

```
python -m marugoto_mods.ViT_extractor \
      --slide_tile_paths SLIDE_TILE_DIR \
      --outdir PATH/TO/SAVE/FEATURES \
      --augmented_repetitions INT
```

### heatmaps
1. Move to the marugoto_mods folder.
2. Activate an appropriate venv.
3. Run the following command:

```
python -m marugoto_mods.create_heatmaps.py [-h] -m MODEL_PATH -o OUTPUT_PATH -t TRUE_CLASS
                                           [--no-pool]
                                           [--mask-threshold THRESH]
                                           [--att-upper-threshold THRESH]
                                           [--att-lower-threshold THRESH]
                                           [--score-threshold THRESH]
                                           [--att-cmap CMAP]
                                           [--score-cmap CMAP]
                                           SLIDE [SLIDE ...]
```
Further information available at https://github.com/KatherLab/highres-WSI-heatmaps 
