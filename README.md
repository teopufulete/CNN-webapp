# CNN-webapp 
CNN picture classifier webapp built in Flask. It differentiates between pictures of cats and dogs.

#### Model arhitecture:
- Input Layer: reshapes image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
- Conv Layer: extracts features from image.
- Pooling Layer: reduces the spatial volume of input image after convolution.
- Fully Connected Layer: connects a layer to another layer.
- Output Layer: makes the prediction.


