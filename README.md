## Reflection Questions

### 1. How does a computer represent a handwritten digit as data?

A computer sees a handwritten digit as a grid of numbers rather than as a number like “5.” Each value represents the brightness of one pixel. In the grayscale images used in my project, 0 is black, 255 is white, and the values in between are shades of gray. The arrangement of those pixels forms the lines and curves that the model learns to recognize.

### 2. Why does the model use a 28 × 28 image?

My model uses 28 × 28 images because that is the size of the MNIST images it was trained on. This gives it 784 pixels per image—enough to show the shape of a digit without making the input unnecessarily large. The app therefore needs to prepare drawings in the same format. I crop the drawing, resize it proportionally to fit inside 20 × 20 pixels, and centre it on a 28 × 28 black background.

### 3. Why are pixel values normalized between 0 and 1?

I divide each pixel value by 255 so that the brightness values fall between 0 and 1. This keeps the inputs on a smaller, consistent scale and helps the network train more reliably as it adjusts its weights. It does not change the shape of the digit; it only changes how its brightness is represented. I apply the same normalization in both the training notebook and the app so the model receives inputs in the format it expects.

### 4. What does the confidence score represent?

The confidence score is the probability the model assigns to its predicted digit. My model's softmax output gives a score for each digit from 0 to 9, and the app displays the highest one as a percentage. For example, if the score for 7 is 0.92 and it is the highest score, the app predicts 7 with 92% confidence. This tells me how strongly the model favours that answer, but it is not a guarantee that the answer is correct—even a confident prediction can be wrong.

### 5. Which type of drawing was hardest for your model to classify, and why do you think that happened?

During testing, I noticed that drawings near the sides of the canvas were harder for the model to classify. I think this was partly because MNIST digits are centred, while my drawings were not always positioned the same way. To address this, I added preprocessing that crops the digit, preserves its proportions, and centres it using its pixel intensities.

I also noticed problems with thin strokes, especially when the pen thickness was below 10. After shrinking a drawing from the canvas to 28 × 28 pixels, those strokes can become faint or lose useful detail. They may also look different from the handwriting the model learned from. The preprocessing preview helps me check whether the image reaching the model still looks like the digit I intended to draw.

### 6. What's the difference between training a model and using a trained model for inference?

Training is when the model learns from examples and their correct labels. In my notebook, it makes predictions on MNIST images, measures its errors, and adjusts its weights to improve. I compare CNNs with 16, 32, and 64 convolution filters, select the best one using validation results, and evaluate it on a separate test set.

Inference is when I use the saved model to classify a new drawing in the app. The app prepares the image and passes it through the model to get a prediction and confidence score. The model's weights do not change during this step, so using the app does not automatically teach it anything new.

### 7. How could you improve this application in the future?

I would add a way for users to save incorrectly classified drawings with the correct labels. I could then review those examples and include them in a future training run, helping the model learn from handwriting that differs from MNIST. Saving the examples alone would not improve the model; I would need to retrain it and check whether its performance actually improved on separate test images.

I would also try training with variations in stroke thickness, position, and rotation to make the model more tolerant of different drawing styles. Another useful improvement would be a low-confidence warning that asks the user to redraw the digit rather than presenting every prediction as equally reliable.
