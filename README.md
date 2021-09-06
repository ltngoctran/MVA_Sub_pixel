# MVA_Sub_pixel

Author: Le Tran Ngoc Tran

Project from the MVA course Sub-pixel Image Processing.

## Packages 
Numpy
Pandas
Matplotlib
Imageio
Scikit-image
Scipy

## Run expeiments
For bilinear interpolation:

```bash
    python -m src.experiments.bilinear_exp
```

For Shannon interpolation:

```bash
    python -m src.experiments.shannon_exp
```

For Shannon interpolation and Gaussian smoothing: run same as Shannon interpolation but uncomment line 78-84

## Resources

| Path | Description
| :--- | :----------
| [Sub-pixel]() | Main folder.
| &boxvr;&nbsp; [data]() | data folder.
| &boxvr;&nbsp; [fit]() | Folder to store csv file containing error for each case.
| &boxvr;&nbsp; [results]() | Store the image results-error map.
| &boxvr;&nbsp; [src]() | the main source codes.
| &boxv;&nbsp; &boxvr;&nbsp; [lib]() | library.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [reductions]() | reduction methods
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [registrations]() | registration methods.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [tools]()   | necessary functions.
| &boxv;&nbsp; &boxvr;&nbsp; [experiments]() | some experiments test cases.
| &boxv;&nbsp; &boxvr;&nbsp; [config.py]() | some configurations.
| &boxvr;&nbsp; [file.csv]() | Save 100 random translation.



