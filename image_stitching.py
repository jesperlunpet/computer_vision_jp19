# import the necessary packages
import numpy as np
from imutils import paths
import imutils
import cv2

MIN_MATCH_COUNT = 10

# Buzz words: Domain specific, 

def featureDetection(img1, img2, detector):
	# FDetect the keypoints with SIFT Detector, compute the descriptors
	if detector.lower() == "sift": 
		fdet = cv2.xfeatures2d.SIFT_create()
	elif detector.lower() == "surf":
		fdet = cv2.xfeatures2d.SURF_create()
	elif detector.lower() == "orb":
		fdet = cv2.ORB_create(nfeatures=500)
		kp0 = fdet.detect(img1, None)
		kp1 = fdet.detect(img2, None)
		kp0, des0 = fdet.compute(img1, kp0)
		kp1, des1 = fdet.compute(img2, kp1)
		# cv2.imshow('original_image_left_keypoints'+str(detector)+".jpg", cv2.drawKeypoints(img1, kp0, None, color=(0,0,255)))
		# cv2.imwrite( "dist/"+str(detector)+".jpg", cv2.drawKeypoints(img1, kp0, None, color=(0,0,255)) );
		return kp0, des0, kp1, des1
	else:
		raise Exception("{} is not a valid input for the detector".format(detector))

	kp0, des0 = fdet.detectAndCompute(img1, None)
	kp1, des1 = fdet.detectAndCompute(img2, None)
	# cv2.imwrite( "dist/"+str(detector)+".jpg", cv2.drawKeypoints(img1, kp0, None, color=(0,0,255)) );
	# cv2.imshow('original_image_left_keypoints'+str(detector)+".jpg", cv2.drawKeypoints(img1, kp0, None, color=(0,0,255)))
	return kp0, des0, kp1, des1

def getMatches(des1,des2, matcher):
	# Matching descriptor vectors with a BF or FLANN based matcher
	match = cv2.DescriptorMatcher_create(matcher)
	matches = match.knnMatch(des1,des2,k=2)
	good = []
	# Filtering the matches
	for m,n in matches:
		if m.distance < 0.4*n.distance:
			good.append(m)
	return good

def combine(img1_, img2_, matcher, detector):
	img1 = cv2.cvtColor(img1_,cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2_,cv2.COLOR_BGR2GRAY)
	kp1, des1, kp2, des2 = featureDetection(img1, img2, detector)
	matches = getMatches(des1, des2, matcher)
	if len(matches) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		h, w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts, M)
		img2 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		dst = cv2.warpPerspective(img1_,M,(img2_.shape[1] + img1_.shape[1], img2_.shape[0]))
		# cv2.imwrite( "dist/warped.jpg", dst) 
		# cv2.imshow("warpPerspective.jpg", dst)
		img3 = cv2.drawMatches(img1_, kp1, img2_, kp2, matches, None, matchColor=(255,0,0), singlePointColor=None, flags=2)
		# cv2.imwrite( "dist/"+str(matcher)+".jpg", img3) 
		# cv2.imshow("original_image_drawMatches"+str(matcher)+".jpg", img3)
		dst[0:img2_.shape[0],0:img2_.shape[1]] = img2_
		# cv2.imshow("original_image_stitched.jpg", dst)
		return dst
	else:
		raise Exception("Not enought sorted matches are found - {}".format(MIN_MATCH_COUNT))

img0_ = cv2.imread("images/IMG_3118.jpg")
img1_ = cv2.imread("images/IMG_3119.jpg")
img2_ = cv2.imread("images/IMG_3120.jpg")

# combine(img0_, img1_, cv2.DESCRIPTOR_MATCHER_FLANNBASED, "sift")
# combine(img0_, img1_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")
img3_ = combine(img0_, img1_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")
# img4_ = combine(img1_, img2_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")
# img5_ = combine(img3_, img4_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")
# img6_ = combine(img0_, img4_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")
# img7_ = combine(img3_, img2_, cv2.DESCRIPTOR_MATCHER_BRUTEFORCE, "sift")

cv2.imshow("in1.jpg", img3_)
# cv2.imwrite( "dist/result1.jpg", img3_) 
# cv2.imshow("final2.jpg", img5_)
# cv2.imwrite( "dist/result2.jpg", img5_) 
# cv2.imshow("final2.jpg", img6_)
# cv2.imwrite( "dist/result3.jpg", img6_) 
# cv2.imshow("in.jpg", img7_)
# cv2.imwrite( "dist/result4.jpg", img7_) 
cv2.waitKey(0)
#cv2.imsave("original_image_stitched_crop.jpg", trim(dst))