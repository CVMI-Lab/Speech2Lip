VIDEO_SEQ=may
SEQ=may

python3 inference.py configs/face_simple_configs/${VIDEO_SEQ}/${SEQ}.yaml \
    --model_path model_may.pt \
    --output_dir ${SEQ}/example \
    --use_new_audio