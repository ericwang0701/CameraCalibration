import cv2
from PIL import Image
import numpy as np
import glob
import os
import pickle
import json
import pandas as pd
import argparse

aruco = cv2.aruco
np.set_printoptions(precision=3)

# Paremeters needed for board detection.
parameters = aruco.DetectorParameters_create()
board_config_filepath = \
    os.path.join(os.getcwd(), '../board/sample_board.csv')
df_board_configs = pd.read_csv(board_config_filepath, index_col=0)
dictionaryID = int(df_board_configs.loc['dict_ID']) # 5: aruco.DICT_5X5_100
dictionary = aruco.getPredefinedDictionary(dictionaryID)
squareL = float(df_board_configs.loc['square_length'])
markerL = float(df_board_configs.loc['marker_length'])
pixels_per_mm = 10 # for checker board image
A4size = (210, 297)
# minimum margin (height, width) when printing in mm
tb = int(df_board_configs.loc['margin_tb'])
tb = int(df_board_configs.loc['margin_lr'])
# unit: mm
squareNumX = int(df_board_configs.loc['num_squares_x'])
squareNumY = int(df_board_configs.loc['num_squares_y'])
boardSizeX = int(df_board_configs.loc['board_size_x'])
boardSizeY = int(df_board_configs.loc['board_size_y'])


def get_file_paths(file_dirpath, file_ext):
    path = os.path.join(file_dirpath, '*.'+file_ext)
    file_names = [os.path.basename(r) for r in glob.glob(path)]
    file_paths = [os.path.join(file_dirpath, fs) \
                    for fs in file_names]
    print(file_names)
    return file_paths, file_names

def get_board_image():
    board = aruco.CharucoBoard_create(squareNumX, 
                                      squareNumY, 
                                      squareL, 
                                      markerL, 
                                      dictionary)
    
    # The third parameter is the (optional) margin in pixels, 
    # so none of the markers are touching the image border.
    # Finally, the size of the marker border, similarly to drawMarker() function. 
    # The default value is 1.
    boardImage = board.draw((boardSizeX*pixels_per_mm, 
                             boardSizeY*pixels_per_mm), 
                             None, 0, 1) # 10 pixels/mm
    return(board, boardImage)

def get_calibration_images(calib_img_paths, resimgs=False):
    calibImages = []
    for calib_img_path in calib_img_paths:
        calibImage = cv2.imread(calib_img_path)
        if calibImage is None:
            print(os.path.basename(calib_img_path)+" cannot be read.")
            continue
            # break
        if resimgs:
            calibImage = cv2.resize(calibImage, (1280,720)) # dont' resize
        calibImages.append(calibImage)
    return calibImages

def show_calibration_result(calibrate_params):
    print("####################")
    retval, cameraMatrix, distCoeffs, rvecs, tvecs, \
        stdDeviationsInstrinsics, stdDeviationsExtrinsics, \
        perViewErrors = calibrate_params
    print("Final re-projection error : \n", retval)
    print("Camera matrix : \n", cameraMatrix)
    print("Vector of distortion coefficients : \n", distCoeffs)
    print("Vector of rotation vectors (see Rodrigues) : \n", rvecs)
    print("Vector of translation vectors : \n", tvecs)
    print("Vector of standard deviations estimated for intrinsic parameters : \n",
        stdDeviationsInstrinsics)
    print("Vector of standard deviations estimated for extrinsic parameters : \n", 
        stdDeviationsExtrinsics)
    print("Vector of average re-projection errors : \n", perViewErrors)

def calibrate_with_ChArUco_board(calibImages, 
                                 param_file_ex='.pkl',
                                 show_calib_result_on_flag=False):
    board, boardImg = get_board_image()

    # detect checker board intersection of ChArUco
    allCharucoCorners = []
    allCharucoIds = []
    charucoCorners, charucoIds = [0,0]
    decimator = 0
    num_images_to_use = 0
    # critetion for sub pixel corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for calImg in calibImages:
        calImg = cv2.cvtColor(calImg, cv2.COLOR_BGR2GRAY) # convert to gray
        # find ArUco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(calImg, dictionary)
        # find ChArUco corners
        if len(corners)>0:
            # sub pixel detection
            for corner in corners:
                cv2.cornerSubPix(calImg, corner,
                                 winSize = (3, 3),
                                 zeroZone = (-1, -1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, calImg, board)
            if (res2[1] is not None) and \
               (res2[2] is not None) and \
               (len(res2[1])>5) and \
               (decimator%decimation_interval==0):
                allCharucoCorners.append(res2[1])
                allCharucoIds.append(res2[2])
                num_images_to_use+=1
            
            decimator+=1
            cv2.aruco.drawDetectedMarkers(calImg, corners, ids)
        # img = cv2.resize(calImg, None, fx=0.5, fy=0.5)
        # cv2.imshow('calibration image',img)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("\n Use "+str(num_images_to_use)+" images for this calibration.")

    try:
        imgSize = calibImages[0].shape[:2]
        calibrate_params = cv2.aruco.calibrateCameraCharucoExtended(
            allCharucoCorners, allCharucoIds, board, imgSize, None, None)
    except:
        print("Failed to calibrate ...")
        print("Not saved.")
        import traceback
        traceback.print_exc()
        return -1

    if show_calib_result_on_flag:
        show_calibration_result(calibrate_params)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, \
        stdDeviationsInstrinsics, \
        stdDeviationsExtrinsics, \
        perViewErrors = calibrate_params
    tmp = [cameraMatrix, \
           distCoeffs, \
           rvecs, tvecs, \
           stdDeviationsInstrinsics, \
           stdDeviationsExtrinsics]

    # save the camera parameters
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)
    
    fpath = os.path.join(calib_result_savedirpath, 
                         calib_result_savename)
    if calib_result_save_format == 'json':
        with open(fpath+'.json', mode='w') as f:
            data = {"camera_matrix": cameraMatrix.tolist(), 
                    "dist_coeff": distCoeffs.tolist(), 
                    "rvecs": rvecs, "tvecs": tvecs}
            json.dump(data, f, sort_keys=True, indent=4, cls=MyEncoder)
    else:
        with open(fpath+'.pkl', mode='wb') as f:
            pickle.dump(tmp, f, protocol=-1)
    
    print("Saved.")
    return 0

def undistort(cam_param_path, images):
    with open(cam_param_path, 'rb') as f:
        camera_params = pickle.load(f)

    cameraMatrix, distCoeffs, rvecs, tvecs, \
        stdDeviationsInstrinsics, \
        stdDeviationsExtrinsics = camera_params

    # write the camera matrix
    imgSize = images[0].shape[:2]
    h,  w = imgSize
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, (w,h), 1, (w,h))
    print(newcameramtx, roi)

    for i, before_undistortImg in enumerate(images):
        dst = cv2.undistort(before_undistortImg, \
                            cameraMatrix, 
                            distCoeffs, 
                            None, newcameramtx)
        
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        if not os.path.isdir(undistort_result_dirpath):
            os.mkdir(undistort_result_dirpath)
        cv2.imwrite(os.path.join(undistort_result_dirpath, \
                                 "undistorted"+str(i+1)+'.png'), 
                    dst)

def board_calibration(undistortion_on_flag=False, 
                      show_calib_result_on_flag=False):

    calib_image_paths, calib_image_names = \
        get_file_paths(calib_image_dirpath, calib_image_format)
    #use default resoultion 1920x1080
    res = calibrate_with_ChArUco_board(
        get_calibration_images(calib_image_paths, 
                               resimgs=False),
        show_calib_result_on_flag=show_calib_result_on_flag)
    
    if res < 0:
        return
    
    if undistortion_on_flag:
        undistort(param_filepath, 
                  get_calibration_images(calib_image_paths))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', type=str, help='calibration images folder')
  parser.add_argument('--image_format', type=str, default='png', help='calibration images format')
  parser.add_argument('--result_format', type=str, default='pkl', choices=['pkl', 'json'], help='calib result format')
  parser.add_argument('--show_result', action='store_true')
  parser.add_argument('--undistortion_on', action='store_true')

  args = parser.parse_args()
  calib_image_dirpath = args.image_dir
  calib_image_format = args.image_format
  calib_result_save_format = args.result_format
  show_calib_result_on = args.show_result
  undistortion_on = args.undistortion_on
  
  #internal settings
  calib_result_savedirpath = os.path.join(os.getcwd(), "../result")
  os.makedirs(calib_result_savedirpath, exist_ok=True)
  calib_result_savename = "camera_param"
  decimation_interval = 2 # 1 means not applied
  undistort_result_dirpath = \
  os.path.join(calib_image_dirpath, "undistort_result/")
  
  param_filepath = os.path.join(os.getcwd(), "../result/camera_param.pkl")
  
  board_calibration(undistortion_on_flag=undistortion_on,
                      show_calib_result_on_flag=show_calib_result_on)