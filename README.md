# Pose Estimation
A simply pose estimation project using Mediapipe

<div>
  MediaPipe Pose is a ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks and background segmentation mask on the whole body from RGB video frames utilizing our BlazePose research that also powers the ML Kit Pose Detection API. Current state-of-the-art approaches rely primarily on powerful desktop environments for inference, whereas our method achieves real-time performance on most modern mobile phones, desktops/laptops, in python and even on the web.
</div><br/>
<div>
  In this project, I decided to combine the Blazepose algorithm with analytic geometry concepts to perform a pose estimation.<br/>
  This work is applied to images, but can be easily adapted to video images.
</div><br/>

<div>
  The algorithm has 33 reference points of the human body, these containing the x,y,z coordinate within the image.<br/>
  <img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png"/>
</div><br/>


<div>
  <li>1. The first step of the algorithm was to obtain the vector AB by subtracting two points. </li>
  <li>2. The second step of the algorithm was to generate the angle formed between two vectors, using the following formula:
    <blockquote>rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))</br>
    graus = degrees(rad)
    </blockquote>
    Through the angles it is possible to have an estimate of the person's pose. For example: Hips with an angle of less than 90° it is very likely that the individual is sitting.
    </li>
   <li>3. The next step was to combine the angles formed to estimate the person's pose, as mentioned in the example.</li>
    
</div>

<div>
</div>

<div>
</div>
