# unrolled convolution
   Vanilia convolutional neural network performs well on Handwritten digit recognition, however, the vanilia
implementation is quite slow. So we follow the following approach to speed up the convolutional arithmetic.
    
 This arithmetic is called unrolled convolution, which converts the processing in each convolutional layer
 (both forward-propagation and back-propagation) into a matrix-matrix product. 
 
 The matrix-matrix product representation of CNNs makes their implementation faster.
 
## reference
1. High Performance Convolutional Neural Networks for Document Processing.  [[Link](https://www.researchgate.net/publication/228344387_High_Performance_Convolutional_Neural_Networks_for_Document_Processing)]
2. Convolution in Caffe. [[Link](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)]