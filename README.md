# :dog2: Stray Dogs Detection System :dog2:


## **I. Introduction**

### **Motivation** 
- High population of stray dogs in India leading to safety concerns and conflicts between humans and stray dogs.
### **Objective**
- Develop a stray dog detection model to identify and locate stray dogs across the city.
### **Solution**
- Create a system using AI and image processing to detect stray dogs and provide their location to authorities.
### **Benefits**
- Efficiently capture stray dogs, enabling authorities to sterilize and vaccinate them, reducing fear and promoting harmony between humans and stray dogs.

## **II. The Dataset**
* Find the pre-processed dataset at [Drive-link](
https://drive.google.com/file/d/1_gsKTzctbe6WZ5oQlpIWW8jy6xXDVRsF/view?usp=sharing

## **III. Model Training**

  * Download the pre-processed dataset from the drive link
  * Upload it on Colab, or your own Drive.
  * If you uploaded it on Drive, then mount your Google Drive in your Colab file, else just upload the dataset's zip file on Colab (it might take some time).
  * Unzip the file using: <br>
     ```cmd
      !unzip <path_to_dataset.zip_file>
     ```
  * Clone Yolov5's GitHub, cd to yolov5 and install the requirements using: <br>
  
     ```cmd
     !git clone https://github.com/ultralytics/yolov5
     %cd yolov5
     %pip install -r requirements.txt
     ```
  * Upload `yolo5s.pt` and `custom_dataset.yaml` on your Colab
  * Train your own model using the command: <br>
     ```cmd
     !python train.py --img 640 --cfg /content/yolov5/models/yolov5m.yaml --hyp /content/yolov5/data/hyps/hyp.scratch-med.yaml --batch 32 --epochs 50 --data /content/custom_dataset.yaml --weights /content/yolov5s.pt  --workers 24 
     ```
     I trained for 100 epochs initially but 90 epochs gave the best result.
  * Save the trained model. You can find it at `/contents/yolov5/runs/train/exp/weights/best.pt`
  * The trained model is the `best.pt` file in this repository.
  * Apply object detection on a video/image file using:
      ```cmd
      !python detect.py --img 640 --source <path_to_source_file> --conf 0.5 --weights /content/yolov5/runs/train/exp/weights/best.pt
      ```
   **Reminder:** you can just use some other dataset, or change the parameters as per your preference to train your own custom object detection model :))

## **IV. Model Deployment Setup**
* This project might only work on your system if you have a GPU (else it will only work for image files and not the video ones)
* You can replace the `best.pt` file with your own custom object detection model.
* cmd:
  ```cmd
   cd <path_to_your_fav_directory>
  ```
   ```cmd
   https://github.com/yashfirkedata/Stray-Dogs-Detection-System-EDI.git
    ```
    ```cmd
    cd <path_to_the_cloned_repository>
    ```
* create 4 folders in `static\` :
  * images
  * predicted_images
  * videos
  * predicted_videos
* create a virtual environment by executing the following command in your cmd
  ```cmd
  python -m venv virtual
  ```
* activate the virtual environment by executing the following command in your cmd
  ```cmd
  virtual\Scripts\Activate.ps1
  ``` 
* install the required libaries by executing the following command in your cmd
  ```cmd
  pip install -r requirements.txt
  ```
* run the following command in your cmd
  ```cmd
  python main-app.py
  ```
* go to `192.168.1.101:5000` on your browser
* choose a file then upload it
* that's it for now, have fun!


**Note: This is an ongoing project, so Ill be updating it from time to time**
<br>
**Also, if you encounter any issue, kindly do let me know so I can fix it!**


