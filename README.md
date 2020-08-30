# image_colorizer_using_CNN

## Best Results:
Input Grayscale Image | Colorized Output Image | Tuned Saturation | Raw Output Color Mask
------------ | -------------  | ------------- | -------------
![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/inputs/1.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/1.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/1_saturated.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/1_mask.jpg)
![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/inputs/5.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/5.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/5_saturated.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/5_mask.jpg)
![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/inputs/6.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/6.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/6_saturated.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/6_mask.jpg)
![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/inputs/4.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/4.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/4.jpg) | ![](https://github.com/mohsenfayyaz/image_colorizer_using_CNN/blob/master/Images/best_results/4_mask.jpg) 


## Comparison with [Zhang](https://arxiv.org/abs/1603.08511)
Please note that as mentiond in [Zhang's paper](https://arxiv.org/abs/1603.08511), their model was trained on over a million color images, however, I only had 10,000 pictures.
Input Grayscale Image | My Result | Increased Saturation | Zhang et al.
------------ | -------------  | ------------- | -------------
![](./Images/inputs/2.jpg) | ![](./Images/best_results/2.jpg) | ![](./Images/best_results/2_saturated.jpg) | ![](./Images/Zhang_results/2.jpg)
![](./Images/inputs/3.jpg) | ![](./Images/best_results/3.jpg) | ![](./Images/best_results/3_saturated.jpg) | ![](./Images/Zhang_results/3.jpg)
