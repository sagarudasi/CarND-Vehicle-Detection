from common import *
import pickle
from moviepy.editor import VideoFileClip
import sys

# Main pipeline function. 
# Takes image and other configuration parameters as input.
# Output is the final image with bounding boxes on car
def pipeline(image, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    windows = []
    windows = windows + slide_window(image, x_start_stop=[100 , (image.shape[1]-100)], y_start_stop=[360, 475], xy_window=(75, 75), xy_overlap=(0.75, 0.75))
    #draw_img = draw_boxes(np.copy(image), windows, color=(0, 255, 0))
    windows = windows + slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[360, 520], xy_window=(100, 100), xy_overlap=(0.75, 0.75))
    # draw_img = draw_boxes(np.copy(image), windows, color=(255, 0, 0))
    windows = windows + slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[380, 630], xy_window=(150, 150), xy_overlap=(0.75, 0.75))
    #draw_img = draw_boxes(np.copy(image), windows, color=(0, 0, 255))
    
    current_car_windows = search_windows(image, windows, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

    # Add heat to each box in box list
    heat = add_heat(heat, current_car_windows)
    heat = apply_threshold(heat,5)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 1)
    # plt.imshow(heatmap, cmap='hot')
    # plt.show()
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    # #draw_img = draw_boxes(np.copy(image), windows, color=(0, 255, 0))

    # #draw_img = draw_boxes(np.copy(image), current_car_windows, color=(0, 255, 0))

    # # draw_img = draw_boxes(image, current_car_windows)
    # # plt.imshow(draw_img)
    # # plt.show()
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    return draw_img


# Read the classifier from the pickle file and the corresponding configuration
data_file = 'ClassifierData.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
    
svc = data['svc'] 
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']
hist_range=data['hist_range']

# Run pipeline on single image as expected by fl_image function
def runpipeline(image):
    global svc, X_scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat
    return pipeline(image, svc, X_scaler,color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# Create a video output from project_video
white_output = 'output_project_video.mp4'
clip1 = VideoFileClip("test_video.mp4")
white_clip = clip1.fl_image(runpipeline)
white_clip.write_videofile(white_output, audio=False)
