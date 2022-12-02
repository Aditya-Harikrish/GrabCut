# GrabCut

## Installing the Dependencies

```
pip install -r requirements.txt
```

## Instructions
1. Specify the path of the image in the above cell
2. Running the below cell, will casue a window to pop up. Press left mouse button to draw a bounding box over the foreground. Release the button and press <kbd> esc </kbd>
3. This will start the GrabCut with only the bounding box input. After it's done, another window will pop up showing the results of the segmentation.
4. Press any key on the keyboard, and a new window with the original image will come up. Compare and see which parts of the image were wrongly segmented and demarcate the foreground and background as in the next step.
5. In the newly popped up window, press left mouse button and draw over areas which are supposed to be foreground; and right mouse button for pixels which are supposed to be background.
6. Press <kbd>esc</kbd>. GrabCut will start again, this with the drawing inputs and the bounding box information.
7. Final result shows up. 

## Images

The images were obtained from [Pexels](https://www.pexels.com) and [Pixabay](https://pixabay.com).