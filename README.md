# Affine-transformation
Find the affine transformation between two channels in a localization microscopy image

Use: 

1. Collect as many FOVs with TetraSpeck as you want, acquire in both colors (500 frames at 25ms is sufficient)
2. Localize these FOVs (Picasso Localize)
3. Load both files into Picasso render, do not align channels!!!, use the pick feature to select circles containing the beads in both channels, save picked localizations for both channels. It needs to be the same picks for both channels, otherwise the numbering will not match up. 
4. Paste the paths to these picked localization .hdf5 files in lines 204-212 in the python file. 
5. To just get the matrix, comment out the final line - instead, print(red_to_green)
6. To apply the matrix to your .hdf5 file of interest, use the final line and add the path to the file you want transformed

Notes:

1. Apply the affine transformation to the .hdf5 files of interest before removing drift in Picasso render. 
2. All localizations from a given channel in a given pick will be averaged - thus it is important to only have one TetraSpeck present per pick.
3. An arbitrary number of FOVs can be used - the more you acquire, the more accurate the transformation will be.