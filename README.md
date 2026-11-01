## Reflection Questions (include in your README)

1.  How does a computer represent a handwritten digit as data?
	As a matrix of numbers. In case of a grey scale image 0 is black and 255 is while and all other numbers represents different shades of grey.
2.  Why does the model use a 28 × 28 image?
	  Efficient enough to fit in numbers and not wasting a lot of computation with big size images which may take more effort in analyze and get features out of an image. 
3.  Why are pixel values normalized between 0 and 1?
      The Neural network computational logic works better with numbers between 0 and 1 than bigger numbers. Weight adjustmets also gets easier. 
4.  What does the confidence score represent?
	 The % of match of an input with what the model had trained with.
5.  Which type of drawing was hardest for your model to classify, and why do you think that happened?
     Initially the drawing on the sides/not centered images where not correctly predicted as the training was done on centred images. So had to add logic to take the drawn image and place it more of a center. Next failure is when the canvas pen width is low like less than 10 it fails. As model was never trained for very thin images. I have added a logic to save those failed images with lable so I can combine those images part of m training set, assuming my model will work better with that.. 
6.  What's the difference between training a model and using a trained model for inference?
     Training a model was with available dataset and once the model is in use for inference it actually try to find pattern matches within the model and comeup with probbale match wth a confidence score and we could set up a minimum threshold above that we can assume the predicted value is correct. 
7.  How could you improve this application in the future?
      As we keep testing the app we will spot images those are not correctly predicted. save those for making part of traiing data the model learns from mistake and continously improve. 