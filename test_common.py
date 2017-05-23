from common import *
from skimage import data, color, exposure

# image = cv2.imread('test_images/test4.jpg')
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(str(image))

# image1 = convert_colorspace(image,'HLS')
# cv2.imshow('image', image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("--------------------------------")
# print(str(image1))

color_space='HLS'
spatial_size=(32, 32)
hist_bins=16
hist_range=(0,255)
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'
spatial_feat=True
hist_feat=True
hog_feat=True

# temp_list = []
# temp_list.append('test_images/test4.jpg')
# # temp_list.append('dataset/final/1.png')
# car_features = multiple_img_features(imgs=temp_list, color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
# print(str(car_features))

# image = cv2.imread('test_images/test4.jpg')
# img_feature = single_img_features(image=image, color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
# print(str(img_feature))


image = cv2.imread('test_images/test4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow('HLS', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# fd, hog_image = hog(image, orientations=orient, 
#                                   pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                   cells_per_block=(cell_per_block, cell_per_block), 
#                                   transform_sqrt=True, 
#                                   visualise=True, feature_vector=False)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_adjustable('box-forced')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# ax1.set_adjustable('box-forced')
# plt.show()