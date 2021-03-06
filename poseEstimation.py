import os
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import degrees


def normalizeVector(vector):
    return vector / np.linalg.norm(vector)

def getAngles(v1, v2):
    ''' 
      Vetor de entrada está no formato nparray
      Exemplo: 
          v1 = np.array([3, 0, 0])
          v2 = np.array([0, 5, 0])
    '''
    v1_u = normalizeVector(v1)
    v2_u = normalizeVector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    graus = degrees(rad)
    return graus

def getVector(pontoA, pontoB):
    vector = np.array([pontoB.x - pontoA.x, pontoB.y - pontoA.y, 0]) #2-dimensional Cartesian plane
    #vector = np.array([pontoB.x - pontoA.x, pontoB.y - pontoA.y, pontoB.z - pontoA.z]) #3-dimensional Cartesian plane
    return vector

def getProportionalDistance(pontoA, pontoB):
    arrA = np.array((pontoA.x, pontoA.y, pontoA.z))
    arrB = np.array((pontoB.x, pontoB.y, pontoB.z))

    dist = np.linalg.norm(arrA-arrB)
    return dist

def calculatePose(ang_LeftHip, ang_RightHip, ang_LeftKnee, ang_RightKnee, ang_LeftShoulder, ang_RightShoulder, ang_LeftElbow, ang_RightElbow):
    
    lowerPose = 'NULL'
    upperLeftPose = 'NULL'
    upperRightPose = 'NULL'

    if ang_LeftHip > 120 and ang_RightHip > 120 and ang_LeftKnee > 120 and ang_RightKnee > 120:
        lowerPose = 'STANDING'
    else:
        lowerPose = 'SITTING'

    if ang_LeftShoulder > 120:
        upperLeftPose = 'RAISED'
    else:
        upperLeftPose = 'DOWN'

    if ang_RightShoulder > 120:
        upperRightPose = 'RAISED'
    else:
        upperRightPose = 'DOWN'

    print('The person is {} with left arm {} and right arm {}'.format(lowerPose, upperLeftPose, upperRightPose))
    return 'null'


def bodyPoints(landmark):
    #Block to identify the relative distance between points
    '''
    dist = getProportionalDistance(landmark[15], landmark[16])
    print('Hands: {}'.format(dist))
    handsDist = 'distant'
    if dist <= 0.25:
        handsDist = 'close'

    dist = getProportionalDistance(landmark[15], landmark[23])
    print('Left hand hip: {}'.format(dist))
    leftHandHipDist = 'distant'
    if dist <= 0.25:
        leftHandHipDist = 'close'

    dist = getProportionalDistance(landmark[16], landmark[24])
    print('Right hand hip: {}'.format(dist))
    rightHandHipDist = 'distant'
    if dist <= 0.25:
        rightHandHipDist = 'close'

    '''
    #Block to identify the angles
    
    #Lower Members
    ##Angle Left Hip
    v23_11 = getVector(landmark[23], landmark[11])
    v23_25 = getVector(landmark[23], landmark[25])
    ang_LeftHip = getAngles(v23_25, v23_11)
    ##Angle Right Hip
    v24_12 = getVector(landmark[24], landmark[12])
    v24_26 = getVector(landmark[24], landmark[26])
    ang_RightHip = getAngles(v24_26, v24_12)
    
    ##Angle Left Knee
    v25_23 = getVector(landmark[25], landmark[23])
    v25_27 = getVector(landmark[25], landmark[27])
    ang_LeftKnee = getAngles(v25_23, v25_27)
    ##Angle Right Knee
    v26_24 = getVector(landmark[26], landmark[24])
    v26_28 = getVector(landmark[26], landmark[28])
    ang_RightKnee = getAngles(v26_24, v26_28)


    #Upper limbs
    ##Angle Left Shoulder
    v11_13 = getVector(landmark[11], landmark[13])
    v11_23 = getVector(landmark[11], landmark[23])
    ang_LeftShoulder = getAngles(v11_13, v11_23)
    ##Angle Right Shoulder
    v12_14 = getVector(landmark[12], landmark[14])
    v12_24 = getVector(landmark[12], landmark[24])
    ang_RightShoulder = getAngles(v12_14, v12_24)
  
    #Angle Left Elbow
    v13_11 = getVector(landmark[13], landmark[11])
    v13_15 = getVector(landmark[13], landmark[15])
    ang_LeftElbow = getAngles(v13_11, v13_15)
    #Angle Right Elbow
    v14_12 = getVector(landmark[14], landmark[12])
    v14_16 = getVector(landmark[14], landmark[16])
    ang_RightElbow = getAngles(v14_12, v14_16)
    
    print('Lower members')
    print('Left Hip: {}'.format(ang_LeftHip))
    print('Right Hip: {}'.format(ang_RightHip))
    print('Left Knee: {}'.format(ang_LeftKnee))
    print('Right Knee: {}'.format(ang_RightKnee))

    print('Upper limbs')
    print('Left Shoulder: {}'.format(ang_LeftShoulder))
    print('Right Shoulder: {}'.format(ang_RightShoulder))
    print('Left Elbow: {}'.format(ang_LeftElbow))
    print('Right Elbow: {}'.format(ang_RightElbow))

    return ang_LeftHip, ang_RightHip, ang_LeftKnee, ang_RightKnee, ang_LeftShoulder, ang_RightShoulder, ang_LeftElbow, ang_RightElbow


def poseEstimation(frame):
    
    position = 'NULL'
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    results = pose.process(frame)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #Exibe as linhas
        ang_LeftHip, ang_RightHip, ang_LeftKnee, ang_RightKnee, ang_LeftShoulder, ang_RightShoulder, ang_LeftElbow, ang_RightElbow = bodyPoints(results.pose_landmarks.landmark)
        position = calculatePose(ang_LeftHip, ang_RightHip, ang_LeftKnee, ang_RightKnee, ang_LeftShoulder, ang_RightShoulder, ang_LeftElbow, ang_RightElbow)
    
    return position

def main():
    path = 'dataImages/img05.jpg'
    image = cv2.imread(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    movimento = poseEstimation(image)
    print(movimento)
    cv2.imshow("Saida", image)
    cv2.waitKey(0)
    

main()