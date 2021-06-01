# Alignment
- use semi-rigid registration
- use pyANTsPy https://github.com/ANTsX/ANTsPy especially the multi-metric registration

# Estimate registration quality
- use the correlation between footprints and centroid like in Sheintuch
- also use the KS2D on centroids (may be even look at it locally)



# Summary of the previous approache

1. image registration to determine the best alignment and rotation of the image 
    - then he aligns the footprints and centroids 
    - then he estimate the quality of the registration with a correlation score on the realigned centroids and footprints
2. calculate all the correlation and distance with neigbour cells (eg distance inf to max_dist ~14 microns)
    - check for neighboring cells and compute corr is the footprints are overlapping

