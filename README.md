# 🚀AIFNet
The open available code for paper "AIFNet: Adaptive Frequency-Channel Interaction and Quad-scale Fusion Network for Multimodal 
UAV Object Detection"
<img width="8458" height="5888" alt="overall" src="fig\AIFNet.png" />

# 📚Dataset
DroneVehicle is a large-scale paired RGB–IR UAV vehicle dataset covering nighttime scenes. Its aerial images contain small, densely distributed and partially occluded vehicles under strong illumination variations, especially at night, which closely matches our RGB–IR UAV small-object fusion detection setting and is suitable for evaluating cross-modal interaction and multi-scale feature fusion. In our experiments, white image borders are removed, each RGB–IR pair is resized to 640×640, and annotations are converted to YOLO horizontal bounding boxes. The processed dataset contains 17,990 training pairs, 1,469 validation pairs.

LLVIP is a paired visible–infrared dataset for low-light pedestrian detection and image fusion. Nevertheless, its strictly aligned RGB–IR image pairs and low-light scenes make it well suited for evaluating the generalization ability of multimodal fusion detection beyond aerial scenarios. It contains aligned 14,665 training pairs and 3,666 test pairs, which are resized to 640×640 for input.

OGSOD-1.0 is a SAR–optical remote-sensing object detection dataset built for optical-guided SAR detection. We use this dataset to verify whether the proposed multimodal interaction and fusion strategy can generalize from RGB–IR fusion to broader cross-modal remote-sensing detection. It contains 14,665 training pairs and 3,666 test pairs, which are resized to 512×512 for input. 

The Dataset used in our paper can be can be obtained from the following link

⭐DroneVehicle:

Link: https://pan.baidu.com/s/1iCMhHJxxL3XCsLeU_CJ27g Code: ssff 

 
⭐LLVIP:

Link: https://pan.baidu.com/s/1GQtg2bGXQEKlDFInlWPQvQ  Code: xjki 


⭐OGSOD-1.0:

Link: https://pan.baidu.com/s/1kho0fNpxK_3PSDmS0Yb-Uw  Code: qfm3 

To evaluate whether the multimodal detector can exploit infrared information when visible observations become unreliable, we generated rain-, sandstorm-, and fog-degraded versions of the RGB images in the original DroneVehicle validation set. Rainy images were produced by reducing brightness and contrast, adding spatially varying atmospheric haze, and compositing multi-scale directional rain streaks. Sandstorm images were generated using warm atmospheric scattering, wavelength-dependent color attenuation, large-scale haze blur, and multi-scale dust particles. Foggy images were synthesized using a spatially varying atmospheric scattering model with non-uniform transmission maps, followed by mild smoothing and desaturation. 

⭐Synthetic adverse-weather DroneVehicle validation set:

Link: https://pan.baidu.com/s/14HYYWDrSNAmZSrsbFXqqQg Code: 9hzj 



# 🔣Model
Our AIFNet weights are already included in the repository and can be found at the following location

🔥DroneVehicle: 

```
\runs\DroneVehicle\train\AIFNet\weights
```
🔥OGSOD-1.0: 
```
\runs\OGSOD\train\512\AIFNet\weights
```
# 🔧Environment
```
conda create -n AIFNet python=3.10

conda activate AIFNet

pip install -U pip setuptools wheel

pip install -r requirements.txt
```

# 🔩Train and test

⚠️Dataset path, GPU, batch size, etc., need to be modified according to different situations.

Train our AIFNet
```
python train.py
```

Test our AIFNet
```
python val.py
```





