# Brain-Segmentation
A machine learning algorithm to segment myelinated white matter (mWM) in infant brain MRIs using Gaussian Mixture Models (GMM). By combining T1-weighted and T2-weighted MRI images as a 2D feature space, the algorithm clusters brain tissues into three classes, distinguishing mWM from grey matter and unmyelinated white matter. The segmentation process includes visualizing joint intensity distributions, predicting posterior probability maps, estimating the proportion of mWM in brain tissue, and analyzing likelihood functions and decision boundaries for classification accuracy. The project includes:
  A Python Script (main.py) that runs the model.
  A Juypter Notebook (Brain Segmentation.ipynb) Task 2 with exploratory data analyis, model training and conclusions about the model at each stage.

# Description
In this project, I aimed to develop a machine learning model for segmenting myelinated white matter (mWM) in infant brain MRIs using Gaussian Mixture Models (GMM). The model leveraged T1-weighted and T2-weighted MRI images as a 2D feature space to classify brain tissues into three categories: myelinated white matter, unmyelinated white matter, and grey matter. To ensure accurate segmentation, I visualized the joint intensity distribution, applied GMM clustering with a fixed random state for reproducibility, and generated posterior probability maps for each tissue class. Further analysis included calculating the average T1 and T2 intensities of mWM, estimating its proportion within brain tissue, and evaluating the likelihood function and decision boundaries. The final implementation involved assessing cluster separation quality and comparing the fitted distributions to the observed joint intensity distribution to refine segmentation accuracy.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal after download image files. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Brain Segmentation.ipynb, download image files and run the cells in Task 2 step by step.

# Images
File: images/T1.p
Description: A pickle file containing the coronal 2D slice of the baby brain MRI.

File: images/T2.p
Description: A pickle file containing same the coronal 2D slice of the baby brain MRI.

# Results
Average T1w intensity of myelinated WM: 852.37
Average T2w intensity of myelinated WM: 680.26
Percentage of brain tissue (excluding CSF) that is myelinated WM: 27.11%
Conclusion: The joint distribution highlights overlapping intensities for T1 and T2, indicating challenges in separating classes. The likelihood map shows concentrated regions but some overlap persists. Decision boundaries effectively partition clusters (blue, green, red) but suggest minor misclassification due to overlap. Individual class distributions fit moderately well but could improve.

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author


