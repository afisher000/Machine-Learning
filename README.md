# Machine Learning Repository
I created this repository to start cleaning/organizing my old machine learning projects so useful code scripts are easy to find. 

## Project Overviews
*Digit Classification*:

The MNIST dataset is a classical starting project for those learning about machine learning. I added my own twist by manually creating a dataset using the OpenCV library to parse images of handwritten digits. Digits are centered/scaled to a 16x16 pixel square. To increase the amount of data, I included rotations and transformations of the digits).

With minimial hyperparameter tuning, it is clear the accuracy saturates somewhere between 98.5-99%. There is also clear confusion between 3s/5s and 4s/9s. Additional training data could target these pairs. 
