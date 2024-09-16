# CrackDetection
Detection of cracks on walls using concepts of image analysis.

----------------------------------------------------------------------------
# Task A – Data Engineering
Data is the fuel for nearly any kind of image-based detection challenge nowadays. If not used for training (i.e.
the learning of suitable model parameters), a dataset must be present at least for evaluating the
performance of a proposed approach. Thus, the acquisition and preparation of data forms a vital step in
designing and developing a detection system.
# a) Data acquisition
# b) Data annotation 
# c) Data split
# d) Data augmentation
# e) Datasets statistics

----------------------------------------------------------------------------
# Task B – Crack Segmentation
In this task you propose an approach to semantically segment the cracks in the image. Semantic
segmentation is the task where every pixel in the input image is assigned a class label in the output. Crack
segmentation is a binary task with the classes no-crack and crack. As shown in Figure 1, it is
recommended to use value 0 for no-crack and 255 for crack.
# a) Thresholding.
# b) Morphological operators will clean up images.
# c) Extract discrete regions by implementing connected component analysis.
# d) Feature engineering.
# e) Classifier: Implemented support vector machines (SVM).

----------------------------------------------------------------------------
# Task C – Crack Analytics
Crack segmentation unfolds its power only when the results undergo further processing. Information such as
the length or the number of branches of a crack are crucial for the assessment of a structure’s condition.
# a) Implement a metric to assess the performance of the implemented approach. 
The standard metric for semantic segmentation is intersection-over-union (IoU). Report the performance of the detector
on the test set and discuss the adequacy of the metric for crack detection.
# b) Thinning in order to reduce the segmentation results to a line-like representation of the crack.
# c) Implement a function to compute the length of the detected cracks in the test set.
# d) Comment on the usefulness of the implemented approach for practical crack detection.
