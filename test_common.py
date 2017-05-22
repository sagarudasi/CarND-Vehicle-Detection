from common import *

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
hist_bins=32
hist_range=(0,1)
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'
spatial_feat=True
hist_feat=True
hog_feat=True

temp_list = []
temp_list.append('test_images/test4.jpg')
# temp_list.append('dataset/final/1.png')
car_features = multiple_img_features(imgs=temp_list, color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
print(str(car_features))

image = cv2.imread('test_images/test4.jpg')
img_feature = single_img_features(image=image, color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
print(str(img_feature))