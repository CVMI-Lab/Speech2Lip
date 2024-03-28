# Speech2Lip

<!-- ![](paper_data/pipeline.png) -->

Official PyTorch implementation for the paper "[Speech2Lip: High-fidelity Speech to Lip Generation by Learning from a Short Video (ICCV 2023)](https://arxiv.org/pdf/2309.04814.pdf)".<br/>
Authors: Xiuzhe Wu, Pengfei Hu, Yang Wu, Xiaoyang Lyu, Yan-Pei Cao, Ying Shan, Wenming Yang, Zhongqian Sun, and Xiaojuan Qi.

Feel free to contact xzwu@eee.hku.hk if you have any questions.

## Prerequisites
- You can create an environment with:
    ```
    pip install -r requirements.txt
    ```
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

    ```
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d/ && pip install -e .
    ```
- [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 
    - Place ```01_MorphableModel.mat``` in ```preprocess/data_util/face_tracking/3DMM/```
    - Convert the file:
        ```
        cd preprocess/data_util/face_tracking/
        python convert_BFM.py
        ```

- [FFmpeg](https://ffmpeg.org/download.html) is required to cut the video and combine the audio with the silent generated videos.

## Data Preprocessing
The source videos used in our experiments are referred to as [LSP](https://github.com/YuanxunLu/LiveSpeechPortraits) and [Youtube Video](https://www.youtube.com/watch?v=K6aTw_VByD0&list=WL&index=2). In this example, we use May's video and provide the bash scripts. After data preprocessing, the training data will be created in the `dataset/may_face_crop_lip/` directory. Please replace it with your own data.

- Video preprocessing
    - Download the original video ```may.mp4```. Refer to [LSP](https://github.com/YuanxunLu/LiveSpeechPortraits) for the URL and duration.
    - Convert to images:
        ```
        ffmpeg -i may.mp4 -q:v 2 -r 25 %05d.jpg
        ```
        - Place the images in ```dataset/may/images/```.
        - Once the data preprocessing is complete, the directory ```dataset/may/``` can be deleted.
    - Extract the audio ```audio.wav```:
        ```
        ffmpeg -i may.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav
        ```
        - Place it in ```dataset/may_face_crop_lip/audio/```.
    - For convenience, we provide the cropped video of May [here](https://connecthkuhk-my.sharepoint.com/:v:/g/personal/xzwu_connect_hku_hk/EYESZCvnrwVApGYGFioXd2sBB_KGmuehwlsiR-SF1qdTAg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=qUZPYF).
- Audio preprocessing
    - Extract the DeepSpeech features ```audio.npy```:
        ```
        cd preprocess/deepspeech_features/
        bash extract_ds_features_may.sh
        ```
        - If successful, a file named ```audio.npy``` will be created in ```dataset/may_face_crop_lip/audio/```.

- Image preprocessing
    - [Only for data preprocessing] Download [```79999_iter.pth```](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) and place it in ```preprocess/face_parsing/```.
    - Generate all the files for training:
        ```
        cd preprocess/
        bash preprocess_may.sh
        ```
- Configuration file
    - We offer a sample in ```configs/face_simple_configs/may/```. 
    - To train with your data, modify the data-related items which are highlighted in the provided sample.

- [Only for train] Sync expert network
    - Download [Sync expert network](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1).
    - Place ```lipsync_expert.pth``` in ```models/```.

## Train Speech2Lip
We use May's video as an example and provide the bash scripts. 
- Train with command:
    ```
    bash scripts/example/train_may.sh
    ```

## Pretrained Models
- Our pretrained models are available [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xzwu_connect_hku_hk/EfeX4SgNRLxMtjqfScSqjTIBxG-d8AmllrQP9arwiSu1sA?e=GFX1Ws).
- To run inference, place the pretrained model ```model_may.pt``` in ```log/face_simple/may```.

## Inference 
We use May's video as an example and provide the bash scripts. 
- For evaluation:
    - Generate images
        ```
        bash scripts/example/inference_may.sh
        ```
        - We split the video into 90% train and 10% test sets.
        - Images are generated in ```rendering_result/may/example/postfusion```.
    - Combine images into a video:
        ```
        ffmpeg -r 25 -i %05d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
        ```
    - Combine the video with the test audio:
        ```
        ffmpeg -i output.mp4 -i audio_test.wav -c:v copy -c:a aac -strict experimental output_with_audio.mp4
        ```
        - For the video demo, split the wav file into 90%/10% using ffmpeg, with the 10% used in inference.
        - We provide ```audio_test.wav``` as an example.
    - Evaluation metrics including PSNR, SSIM, CPBD, LMD and Sync score can be applied.
- For any given audio:
    - Place new audio ```audio.npy``` in ```dataset/may_face_crop_lip/audio_test/```
    ```
    bash scripts/example/inference_new_audio_may.sh
    ```

## Citation
If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{wu2023speech2lip,
  title={Speech2Lip: High-fidelity Speech to Lip Generation by Learning from a Short Video},
  author={Wu, Xiuzhe and Hu, Pengfei and Wu, Yang and Lyu, Xiaoyang and Cao, Yan-Pei and Shan, Ying and Yang, Wenming and Sun, Zhongqian and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22168--22177},
  year={2023}
}
```

## Acknowledgments
We use [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) to compute head mask in the canonical space, [DeepSpeech](https://github.com/mozilla/DeepSpeech) for audio feature extraction, [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) for sync expert network, and we are highly grateful to [ADNeRF](https://github.com/YudongGuo/AD-NeRF) for their data preprocessing script.
