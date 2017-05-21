from common import *
import pickle

def pipeline(image, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # plt.imshow(image)
    # plt.show()
    # window_list[0] = []
    # window_list[1] = []
    current_windows_1 = slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[410, 480], xy_window=(70, 70), xy_overlap=(0.5, 0.5))
    # timage = draw_boxes(image, current_windows, color=(255, 0, 0))
    current_windows_2 = slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[400, 520], xy_window=(120, 120), xy_overlap=(0.5, 0.5))
    # current_windows.append(current_windows_1)
    current_windows_3 = slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[380, (380+180)], xy_window=(180, 180), xy_overlap=(0.5, 0.5))
    # current_windows.append(current_windows_2)
    current_windows_4 = slide_window(image, x_start_stop=[0 , image.shape[1]], y_start_stop=[380, (380+240)], xy_window=(240, 240), xy_overlap=(0.5, 0.5))
    window_list = current_windows_1 + current_windows_2 + current_windows_3 + current_windows_4
    #timage = draw_boxes(image, window_list, color=(0, 255, 0))

    #return timage, heat

    current_car_windows = search_windows(image, window_list, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

    # Add heat to each box in box list
    heat = add_heat(heat, current_car_windows)
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 1)
    # plt.imshow(heatmap, cmap='hot')
    # plt.show()
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    # draw_img = draw_boxes(image, current_car_windows)
    # plt.imshow(draw_img)
    # plt.show()
    return draw_img


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

hist_range=(0,1)

# test_image = mpimg.imread('./test_images/test1.jpg')
# prevheat = np.zeros_like(test_image[:,:,0]).astype(np.float)
# final_image, heat = pipeline(test_image, prevheat, svc, X_scaler,color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
# plt.imshow(final_image)
# plt.show()
# # pipeline(np.true_divide(mpimg.imread('./test_images/test2.jpg'), 255), svc, X_scaler)
# # pipeline(np.true_divide(mpimg.imread('./test_images/test3.jpg'), 255), svc, X_scaler)
# test_image = mpimg.imread('./test_images/test4.jpg')
# prevheat = np.zeros_like(test_image[:,:,0]).astype(np.float)
# final_image, heat = pipeline(test_image, prevheat, svc, X_scaler,color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

# plt.imshow(final_image)
# plt.show()

# pipeline(np.true_divide(mpimg.imread('./test_images/test5.jpg'), 255), svc, X_scaler)
# pipeline(np.true_divide(mpimg.imread('./test_images/test6.jpg'), 255), svc, X_scaler)

videofile = cv2.VideoCapture('test_video.mp4')
#video = cv2.VideoWriter('output.avi',-1, 1, (img.shape[1],img.shape[0]))

i=-1
heat = None
prevheat = None
while(videofile.isOpened()):
    ret, frame = videofile.read()
    if ret == True:
        i = i + 1
        if prevheat == None:
            prevheat = np.zeros_like(frame[:,:,0]).astype(np.float)
        result = pipeline(frame, svc, X_scaler,color_space=color_space, spatial_size=spatial_size,hist_bins=hist_bins, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        cv2.imshow('frame',result)
        # prevheat = heat
        #video.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#video.release()
videofile.release()
cv2.destroyAllWindows()