



The Captcha image consists of four characters that are either a string
of letters or a combination of letters with numbers. In some of the
combinations, the captcha characters contain all letters but they are
not aligned cleanly. Some of the letters are joined together or
diagonally aligned, therefore correctly separating them in order to make
them easily recognised as letters is one of the tasks at hand. In the
instances of Captcha characters that contain both numbers and letters,
the position of the number is not guaranteed and the letters are
slanted.

The expected outcome is to correctly identify each of the Captcha
characters from the image. We are doing this inorder to measure the
efficiency of the neural networks to correctly identify the Captcha
characters. The label variable is the letter or number from the image of
the separated Captcha characters.



``` {.python}
!pip install opencv
conda install -c conda-forge/label/gcc7 opencv
conda install -c menpo opencv

# Common imports
import numpy as np
import os


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
```

``` {.python}
 import os
```

``` {.python}
os.getcwd()
```

``` {.python}
#image_file_Path ="/home/LC/chakth01/Neural networks/targetdir/captcha_images/"
image_file_Path ="/home/LC/mbuako01/DS 420 Project 2/imageFolder/captcha_images/"
```

``` {.python}
image_file_Path
```

``` {.python}
def read_image(image_file_path):
    """Read in an image file."""
    bgr_img = cv2.imread(image_file_path)
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    
    return rgb_img 
```

``` {.python}
import cv2
import imutils
import numpy as np
import os
from imutils import paths
import pandas as pd
```

``` {.python}
images = []
labels = []
```

``` {.python}
for image_file_path in imutils.paths.list_images(image_file_Path):
    image_file = read_image(image_file_path)
    label = image_file_path.split('/')[7]
    images.append(image_file)
    labels.append(label)
    
```

``` {.python}
#images
```

``` {.python}
#labels
```

``` {.python}
newLabels = []

for label in labels:
    labelHere = label.split('.')[0]
    newLabels.append(labelHere)
```

``` {.python}
#newLabels
```

``` {.python}
images = np.array(images)
#images4Plot = np.array(images, dtype="float") / 255.0
labels = np.array(newLabels)
```

``` {.python}
print(labels)
```


``` {.python}
#print(images)
```

A dataset of 9,955 of unique CAPTCHA images each with its label as the
filename was used for this research. However, machine learning
classification requires a one-to-many relationship between a label and
in this context the CAPTCHA images. Therefore, uniqueness of the CAPTCHA
images is problematic for a machine learning process.

``` {.python}
images.shape
```


``` {.python}
some_digit = images[300]
#Some_digit_image = some_digit.reshape(24, 72, 3)
plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/d0de6d4167aae20e07f8c9576a487b463410295a.png)

``` {.python}
labels[300]
```


``` {.python}
import os
import os.path
import cv2
import glob
import imutils
```

``` {.python}
def pureBlackWhiteConversionThreshold(image):
    # Add some extra padding around the image
    imagePadded = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(imagePadded, cv2.COLOR_RGB2GRAY)
    # threshold the image (convert it to pure black and white)
    imagethresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    
    return imagethresholded 
```


``` {.python}
def pureBlackWhiteConversionOGImage(image):
    # Add some extra padding around the image
    imagePadded = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(imagePadded, cv2.COLOR_RGB2GRAY)
       
    return gray 
```

``` {.python}
padded_ThreshImage300 = pureBlackWhiteConversionThreshold(images[300])
```

``` {.python}
some_digit = padded_ThreshImage300

plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/8995982a1771d8fa3139a964a1f0de4b7efcd475.png)

``` {.python}
def regionsOfLetters(image):
    
     # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []
    
    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))
    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    #if len(letter_image_regions) != 4:
       # continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    return letter_image_regions 
```

``` {.python}
letter_image_regions = regionsOfLetters(padded_ThreshImage300)
letter_image_regions
```


To deal with the uniqueness problem of the dataset, the solution was to
separate the CAPTCHA images into the individual 4 characters that make
up the CHAPTCHA image. This was to make each character into its own
image. The resulting dataset has 39,754 images with one character per
image. The new dataset satisfies the one-to-many relationship between
the images and the following 32 characters labels {\'2\', \'3\', \'4\',
\'5\', \'6\', \'7\', \'8\', \'9\', \'A\', \'B\', \'C\', \'D\', \'E\',
\'F\', \'G\', \'H\', \'J\', \'K\', \'L\', \'M\', \'N\', \'P\', \'Q\',
\'R\', \'S\', \'T\', \'U\', \'V\', \'W\', \'X\', \'Y\', \'Z\'}

``` {.python}
def extractLetters(letter_image_regions, image):
    # Save out each letter as a single image
    letter_images =[]
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        #image_file1 = read_image(letter_image)
        letter_images.append(letter_image)
    return letter_images 
    
```

``` {.python}
grayScaleImage = pureBlackWhiteConversionOGImage(images[300])
letter_image_List = extractLetters(letter_image_regions,grayScaleImage)
```

``` {.python}
checkImage = letter_image_List[3]
```

``` {.python}
some_digit = checkImage
#Some_digit_image = some_digit.reshape(24, 72, 3)
plt.imshow(some_digit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/2492e78e7724e2c4241d4148b15603de796bd78a.png)

``` {.python}
len(labels)
```


``` {.python}
len(images)
```

``` {.python}
images.shape
```


``` {.python}
def expand2square(image):
    desired_size = 28
    im = image
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im
```

``` {.python}
letterImages = []
letterImageLabels = []


# loop over the unseparated image list
for label, image in zip(labels, images):
    
    padded_ThreshImage300 = pureBlackWhiteConversionThreshold(image)
    letter_image_regions = regionsOfLetters(padded_ThreshImage300)
    grayScaleImage = pureBlackWhiteConversionOGImage(image)
    letter_image_List = extractLetters(letter_image_regions,grayScaleImage)
     #image_reshape = letter_bounding_box.reshape(L0, L1)
    for letter_bounding_box, letter_text in zip(letter_image_List, label): 
        L0=letter_bounding_box.shape[0]
        L1=letter_bounding_box.shape[1]
       
        imageResize = expand2square(letter_bounding_box)
        letterImages.append(imageResize)
        letterImageLabels.append(letter_text)
    
        
        
```

``` {.python}
len(letterImageLabels)
```

``` {.python}
displayDigit = letterImages[609]

plt.imshow(displayDigit, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/ee0447a3ff48b1fb5e393400494480c4aaf221bf.png)

``` {.python}
letterImages[609].shape
```

``` {.python}
images = np.array(letterImages)
labels = np.array(letterImageLabels)
```

``` {.python}
 images.shape
```


``` {.python}
def plot_digit(image):
    some_digit = image
    plt.imshow(some_digit, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


    plt.show()
```

``` {.python}
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```

``` {.python}
index, = np.where(labels == 'F')
index
```


``` {.python}
plt.figure(figsize=(15, 15))
example_images = np.r_[images[[14,39,51,39702,39752]], 
                       images[[7,56,61,39703,39714]],
                       images[[45,   198,   352,39705, 39719]], 
                       images[[2,    12,    52, 39698, 39712]], 
                       images[[3,    26,    87, 39612, 39619]]]

example_images
plot_digits(example_images, images_per_row=5)
#save_fig("more_digits_plot")
#plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/deb47268075c5456f794f88455f6f8a4307c4491.png)

## 4. Prepare Data for RandomForest Model

``` {.python}
images3DTo2D = images.reshape(39754, 28 * 28)
images3DTo2D.shape
```


``` {.python}
X, y = images3DTo2D, labels
X.shape
```


``` {.python}
np.unique(y)
```




## 5. Train test split

The image data was split into training and test set data, with the
training set having 80% of the data and the remaining 20% was kept for
the evaluation of model performance on the test set. The data was then
fit to the random forest model.

``` {.python}
from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=11
)
```

``` {.python}
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
```

``` {.python}
forest_clf.fit(X_train, y_train)
```


### Random Forest training set Performance

``` {.python}
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
```

``` {.python}
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

```

``` {.python}
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
```

``` {.python}
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
#%matplotlib inline


# Binarize the output
y_trainBinarize = label_binarize(y_train, classes=np.unique(y))
n_classes = y_trainBinarize.shape[1]
n_classes

clf = OneVsRestClassifier(forest_clf)
clf.fit(X_train, y_trainBinarize)

y_score = clf.predict_proba(X_train)
```

``` {.python}
y_train_pred = cross_val_predict(clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```


``` {.python}
plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/5365f1f1c87b6a653510fb09a5bb997364b04f31.png)

``` {.python}
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```

``` {.python}
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/778e962171f372a179601b5905b0c84a4ebe1885.png)

``` {.python}
accuracy_score(y_train, y_train_pred)
```


``` {.python}
precision_score(y_train, y_train_pred, average='micro')
```


``` {.python}
recall_score(y_train, y_train_pred, average='micro')
```


``` {.python}
f1_score(y_train, y_train_pred, average='micro')
```


``` {.python}
# precision recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_trainBinarize[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2)
    
plt.xlabel("recall")
plt.ylabel("precision")
#plt.legend(loc="upper left")
plt.title("precision vs. recall curve")
plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/6216f1e130aafc561c1cb81b4dc53af4c6351c18.png)

### Training set ROC curves

``` {.python}
# roc curve
fpr = dict()
tpr = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_trainBinarize[:, i],
                                  (y_score[:, i]))
    plt.plot(fpr[i], tpr[i], lw=2)

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")

plt.title("ROC curve")
plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/c4f0309f843c8a99a6d4b59884e7874273bc8c20.png)

The random forest model did very well predicting the individual
characters from the CAPTCHA images with an accuracy score, precision
score, recall score and f1 score, all of 97.956%. The precision vs
recall and ROC curves shows that the model is predicting most characters
correctly.

### Random Forest test set Performance

``` {.python}
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
y_test_pred = cross_val_predict(clf, X_test_scaled, y_test, cv=3)
conf_mx = confusion_matrix(y_test, y_test_pred)
conf_mx
```


``` {.python}
plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/02ed9d25de4863f789686ec3e9ab332e769bdbc3.png)

``` {.python}
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```

``` {.python}
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/ed5faee1a695e195ac14ddef46077aef664a31bc.png)

``` {.python}
accuracy_score(y_test, y_test_pred)
```


``` {.python}
precision_score(y_test, y_test_pred, average='micro')
```


``` {.python}
recall_score(y_test, y_test_pred, average='micro')
```


``` {.python}
f1_score(y_test, y_test_pred, average='micro')
```


``` {.python}
y_testBinarize = label_binarize(y_test, classes=np.unique(y))
y_scoretest = clf.predict_proba(X_test)
```

``` {.python}
# precision recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_testBinarize[:, i],
                                                        y_scoretest[:, i])
    plt.plot(recall[i], precision[i], lw=2)
    
plt.xlabel("recall")
plt.ylabel("precision")
#plt.legend(loc="upper left")
plt.title("precision vs. recall curve")
plt.show()
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/1826b54ce935dbed15f5aa627b9ea4279e96d1b5.png)

``` {.python}
# roc curve
fpr = dict()
tpr = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_testBinarize[:, i],
                                  (y_scoretest[:, i]))
    plt.plot(fpr[i], tpr[i], lw=2)

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")

plt.title("ROC curve")
plt.show()
```

![](vertopal_506752d0a93e4361b07a09574f75cb2c/3abb83cff29d6b08e08d7c2d2f80828f72bbf3ea.png)

The model's performance on the test is not that much different from the
training set which shows that the model didn't overfit on the training
set. All the performance metrics are above 97% and the precision vs
recall and ROC curves confirm the performance of the model. The
confusion matrix shows that the model is confusing some Ms for Ws and
vice-versa.

``` {.python}
uniqueCharacters = np.unique(y)
uniqueCharacters
```


``` {.python}
W = uniqueCharacters[28]
```

``` {.python}
M = uniqueCharacters[19]
```

``` {.python}
indexM, = np.where(labels == M)
indexM
```


``` {.python}
indexW, = np.where(labels == W)
indexW
```


``` {.python}
plot_digit(images[indexM[789]])
```

``` {.python}
plot_digit(images[indexW[789]])
```

![](vertopal_506752d0a93e4361b07a09574f75cb2c/5935341817aec8ba2aa6411e7cee056d99ea662d.png)

``` {.python}
cl_a = 'M'
cl_b = 'W'

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]

```

``` {.python}
 plot_digits(X_aa[:24], images_per_row=4)
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/c44d061bc67695fa2de68f01fbfb82dd4c515fb0.png)

``` {.python}
plot_digits(X_ab, images_per_row=7)
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/aa188acd9d1d644310939300bd1cd6a7080800ed.png)

``` {.python}
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
```

``` {.python}
plot_digits(X_bb[:25], images_per_row=5)
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/4e9644350b60a39f90935e066d72bb0f377223fc.png)

``` {.python}
plot_digits(X_ba, images_per_row=5)
```


![](vertopal_506752d0a93e4361b07a09574f75cb2c/c3e53611170b82e1d438cf9942a35ccae58a116b.png)

