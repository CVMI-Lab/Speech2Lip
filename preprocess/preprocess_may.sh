export CUDA_VISIBLE_DEVICES=0
ORI_SEQ=may
SEQ=may_face_crop
H=80
W=120
BASEDIR=../dataset/${SEQ}_lip

# STEP0: crop face image. The last four parameters are set based on the source video.
INPUT=../dataset/${ORI_SEQ}/images
OUTPUT=${BASEDIR}/ori_images_face
python3 crop_face.py --img_path ${INPUT} --res_folder ${OUTPUT} \
    --dst_face_h 500 --dst_face_w 500 --face_center_point_x 930 --face_center_point_y 275

# STEP1: detect landmark and facial bounding box
INPUT=$BASEDIR/ori_images_face
OUTPUT=$BASEDIR/landmarks
python3 detect_landmarks.py ${INPUT} ${OUTPUT}

# STEP2: compute 3DMM parameters
INPUT=${BASEDIR}/ori_images_face
python3 face_tracker.py \
    --idname ${SEQ} \
    --id_dir ${BASEDIR} \
    --input ${INPUT} \
    --func 'compute_3dmm'

# STEP3: warp facial images to a canonical space to enable further cropping of the lip region
INPUT=$BASEDIR/ori_images_face
python3 face_tracker.py \
    --idname ${SEQ} \
    --id_dir ${BASEDIR} \
    --input ${INPUT} \
    --func 'warp_image'

# STEP4: compute pixel correspondence for training and compute face masks
INPUT=$BASEDIR/ori_images_face
python3 face_tracker.py \
    --idname ${SEQ} \
    --id_dir ${BASEDIR} \
    --input ${INPUT} \
    --func 'compute_uv_mapping'

# STEP5: compute head mask, face mask for the canonical space
INPUT=$BASEDIR/ori_images_face
python3 face_tracker.py \
    --idname ${SEQ} \
    --id_dir ${BASEDIR} \
    --input ${INPUT} \
    --func 'compute_canonical_mask'

# STEP6: crop lip
INPUT=$BASEDIR/ori_images_face
python3 face_tracker.py \
    --idname ${SEQ} \
    --id_dir ${BASEDIR} \
    --input ${INPUT} \
    --func 'crop_lip' \
    --dst_mouth_h ${H} --dst_mouth_w ${W} 