# Grayscale to RGB

Picking foraminifera samples is as important as tedious for bio-marine research.
Given the fact that it is suitable for automation, our goal was to improve accuracy of those automated
methods thanks to ways of processing images.
Multiple new strategies were implemented: Averaged percentile, Gaussian, Clustering, Weighted
Clustering.
Starting from 16 different images of the same sample we used them to create a single RGB image to
later train a CNN and compare our new accuracy results with the one originally implemented.
Even though our methods were not as accurate as Percentile by themselves, using ensemble strategies
we were able to marginally increase the overall accuracy.

You can read <a href=".\Relazione.pdf">Relazione.pdf</a> for further information about our project.
