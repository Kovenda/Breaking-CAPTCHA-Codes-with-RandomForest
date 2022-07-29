# Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

## 1. Prepare Data for RandomForest Model

images3DTo2D = images.reshape(39754, 28 * 28 #images is from the imageprocessing.py file in this repository
images3DTo2D.shape

X, y = images3DTo2D, labels
X.shape

np.unique(y)

## 2. Train test split
# The image data was split into training and test set data, with the training set having 80% of the data and the remaining 20% was kept for the evaluation of model performance on the test set. The data was then fit to the random forest model.

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=11
)

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

forest_clf.fit(X_train, y_train)

### Random Forest training set Performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

def plot_confusion_matrix(matrix):
    #If you prefer color and a colorbar
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

# Binarize the output
y_trainBinarize = label_binarize(y_train, classes=np.unique(y))
n_classes = y_trainBinarize.shape[1]
n_classes

clf = OneVsRestClassifier(forest_clf)
clf.fit(X_train, y_trainBinarize)

y_score = clf.predict_proba(X_train)

y_train_pred = cross_val_predict(clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()

accuracy_score(y_train, y_train_pred)

precision_score(y_train, y_train_pred, average='micro')

recall_score(y_train, y_train_pred, average='micro')

f1_score(y_train, y_train_pred, average='micro')

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

"""### Training set ROC curves """

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

## The random forest model did very well predicting the individual characters from the CAPTCHA images with an accuracy score, precision score, recall score and f1 score, all of 97.956%. The precision vs recall and ROC curves shows that the model is predicting most characters correctly.
### Random Forest test set Performance

X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))
y_test_pred = cross_val_predict(clf, X_test_scaled, y_test, cv=3)
conf_mx = confusion_matrix(y_test, y_test_pred)
conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()

accuracy_score(y_test, y_test_pred)

precision_score(y_test, y_test_pred, average='micro')

recall_score(y_test, y_test_pred, average='micro')

f1_score(y_test, y_test_pred, average='micro')

y_testBinarize = label_binarize(y_test, classes=np.unique(y))
y_scoretest = clf.predict_proba(X_test)

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
#The model’s performance on the test is not that much different from the training set which shows that the model didn’t overfit on the training set. All the performance metrics are above 97% and the precision vs recall and ROC curves confirm the performance of the model. The confusion matrix shows that the model is confusing some Ms for Ws and vice-versa.


uniqueCharacters = np.unique(y)
uniqueCharacters

W = uniqueCharacters[28]

M = uniqueCharacters[19]

indexM, = np.where(labels == M)
indexM

indexW, = np.where(labels == W)
indexW

plot_digit(images[indexM[789]])

plot_digit(images[indexW[789]])

cl_a = 'M'
cl_b = 'W'

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]

plot_digits(X_aa[:24], images_per_row=4)

plot_digits(X_ab, images_per_row=7)

X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]

plot_digits(X_bb[:25], images_per_row=5)

plot_digits(X_ba, images_per_row=5)

