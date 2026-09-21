# Local data

The MRI/DICOM images and reference masks used in the manuscript are not included in this repository.

The current experiment script expects the following local structure:

```text
data/
├── ST003-SE005-MR022
├── ST003-SE007-MR018
├── ST004-SE007-MR021
└── PixelLabelData/
    └── pixelLabelData/
        ├── Label_1_ST003-SE005-MR0022_MRI.png
        ├── Label_2_ST003-SE007-MR0018_MRI.png
        └── Label_3_ST004-SE007-MR0021_MRI.png
```

The three PNG files are the manually delineated **reference masks** used only for evaluation. They are not used to fit the clustering models or to select the liver cluster.

If redistribution rights for the image data are established later, the data-access instructions can be updated here.
