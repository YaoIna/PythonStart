from deep_learning import lr_utils
import numpy as np

from deep_learning.logistic_regression.cat_model import CatModel

train_x_orig, train_y, test_x_orig, test_y, classes = lr_utils.load_data_set()
model = CatModel()
model.fit(train_x_orig, train_y=train_y, num_iteration=2000, learning_rate=0.005, print_cost=True)
prediction = model.predict(test_x_orig)
print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction - test_y)) * 100))

# ## START CODE HERE ## (PUT YOUR IMAGE NAME)
# my_image = "my_image.jpg"  # change this to the name of your image file
# ## END CODE HERE ##
#
# # We preprocess the image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
#
# plt.imshow(image)
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
#     int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
