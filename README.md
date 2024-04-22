# av_papers_collection
the repo for paper collections for AV related



## General CNN

* Speeding up convolutional networks with Low Rank Expansions (2014, Oxford)
    * exploring filter redundancy to construct a low-rank basis of filters that are rank-1 in the spatial domain 

* cuDNN: efficient Primitives for Deep Learning (2014, Nvidia)

* ResNet: Deep Residual Learning for Image Recognition (2015, Kaiming)

* Fast Algorithms for Convolutional Neural Networks (2015)
    * Winograd for small kernels
    * FFT for large kernels

* Delving Deep into Rectifilers: surpassing Human-level performance on ImageNet (2015 kaiming he)
    * proposed Parmetric Rectified Linear Unit(PReLU)

* Spatial Pyramid Pooling in Deep Convolution Networks for visual recognition (2015 Kaiming)

* Faster R-CNN: towards real-time object detection with region proposal networks (Shaoqing 2016)

* Object Detection Networks on Convolutional Feature Maps (Shaoqing, Kaiming, 2016) 

* Training region-based object detectors with online hard example mining (FAIR, 2016)

* Multi-scale context aggregation by dilated convolutions (2016, Intel)

* SSD: Single Shot Multibox detector (2016)

* YOLO-v1: you only look once, unified, real-time object detection (2016)

* Yolo-v2 (YOLO9000): better, faster and stronger (2016)

* FPN: Feature Pyradmid Networks for object detection (FAIR/Kaiming 2017)

* Understanding the effective receptive field in Deep Convolutional Neural Networks (2017, Toronto)

* Sparsity Invariant CNNs (2017)

* Focal Loss for dense object detection (FAIR/Kaiming, 2018)

* An alaysis of scale invariance in object detection by Scale Normalization Image Pyramid(SNIP) (2018)

* MegDet: a large mini-batch object detector (Megvvi, 2018)

* YoloV3: An incremental improvement (2018)

* DetNet: a backbone network for object detection(Megvii 2018)

* DetNas: backbone search for object detection (Megvii 2019)

* Region Proposal by Guided Anchoring (2019 SenseTime)

* LFIP-SSD: Efficient Featurized Image Pyramid Network for single shot detector (2019 CVPR)

* VovNet: an energy and GPU-computation efficient backbone network for real-time object detection(2019 CVPR)

* Libra R-CNN: towards balanced learning for object detections(SenseTime, 2019 CVPR)

* NAS-FPN: Learning scalable feature pyramid architecture for object detection (Google Brain, 2019 CVPR)

* Understand Geometry of encoder and decoder of CNNs (2019)

* YOLACT: real-time instance segmentation (2019)

* A survey of deep learning based object detection (2019)

* On netwrok design spaces for visual recognition (2019, FAIR)

* Imbalance Problems in object detection: A Review (2020)

* AutoAssign: differentiable label assignement for dense object Detection(Megvii 2020) 

* Large-Scale object detection in the wild from imbalanced multi-labels (SenseTime, 2020)

* EfficientDet: scalable and efficient object detection (Google Brain, 2020)

* Yolov4: optimal speed and accuracy of object detections (2020)

* RegNet: Designing Network Design Spaces (2020, FAIR)

* GhostNet: more features from cheap operations (2020, Huawei)

* Point-GNN: graph neural network for 3D Object Detection in point clouds (CMU, 2021 CVPR)

* RepVGG making VGG-style ConvNets Great Again (Megvii 2021)

* Sample-Free: is Heuristic sampling necessary in training object detectors (2021)

* SOLQ: Segmenting Objects by Learning Queries (Megvii, 2021)

* Yolox: exceeding Yolo series in 2021 (2021)

* Florence, a new foundation model for CV (2021)

* YoloP: you only look once for panoptic driving perception (2022)
    * multi-task

* YoloP-v2: better, faster and stronger for panoptic driving perception (2022)

* a ConvNet for the 2020s (2022, FAIR)

* TorchSparse: efficient pont cloud inference engine (2022, MIT)


## light-weight CNNs

* Sparse Convolutional Neural Networks (2015)
    * sparsity by exploiting inter/intra channel redundancy with a fine-tuning step to minimize accuracy loss

* SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size(2016, Berkeley)

* Speed/accuracy trade-offs for modern convolutional object detectors (Google, 2017)

* The power of sparsity in convolutional neural networks (Google, 2017)
    * a strategy for deactivating connections between filters to save time and memory

* Light-Head R-CNN: in defense of two-stage object detectors (2017, Megvii)

* Pruning Filters for efficient convnets (2017)
    * pruning filters that have small effects on output accuracy 

* Learning efficient convolutional networks through network slimming (2017, Intel)

* ShuffleNet-v1: an extremely efficient convolutional neural network for mobile devices (2017, Megvii)

* Xception: deep learning with depthwise separable convolutions (2017, Google)

* MobileNet-v1: efficient convolutional neural networks for mobile vision applications (2017, Google)

* ShuffleNet-v2: practical guidelines for efficient CNN architecture design(2018, Megvii)

* Recent Advances in efficient computatoin of deep convolutional neural networks: A review (2018)
    * acceleration in algorithm and hardware design

* Efficient Deep Learning inference based on model compression (2018, Alibaba)
    
* MobileNet-v2: inverted residuals and linear bottlenecks (2019, Google)

* SENet: Squeeze-and-Excitation networks (2019, Momenta )

* FCOS:  fully convolutional one-stage object detection (2019)

* Searching for MobileNetV3 (2019, Google)

* MnasNet: platform-aware neural architecture search for Mobile(2019, Google)

* A survey of model compression and acceleration for deep neural networks (2020)

* EfficientNet-v1: rethinking model scaling for convolutional neural networks (2020, Google)

* EfficientNet-v2: smaller models and faster training (2021, Google)

* object detection made simpler by eliminating heuristic NMS (2021, Alibaba)
    * NMS-free by employing a stop-gradient operation 

* VIT-Slim:  multi-dimension searching in continuous optimization space (Berkeley, 2022 CVPR)

## self-supervised & CV Foundational Models 

* self-training with noisy student improved imageNet classification (2020, Google)
    * first train EfficentNet on labeled images, and use it as teacher to generate pseudo labels for 300M unlablled images, then train a larger EfficientNet as student model on the combination of labeled and pseudo labeled images. 

* SimCLR: a simple framework for contrastive learning of visual representation (2020, Google)

* MoCo: momentum contrast for unsupervised visual representation learning (2020, FAIR)

* PIC: parametric instance classification for unsupervised visual feature learning (2020, Microsoft)

* towards open world object detection (2021)

* DetCo: unsupervised contrastive learning for object detection (2021, UHK)

* BeiT: BERT pre-training of image transformers (2021, Microsoft)

* Benchmarking detection transfer learning with vision transformers (2021, FAIR)
    * observed mask-based method(BEIT, MAE) show considerable gains over both supervised and random initialization and these gains increase as model size increases.

* DINO: emerging properties in self-supervised vision transformers (2021, FAIR) 
    * evidence that self-supervised learning could be the key to develper BERT-like model based on VIT 

* MAE: masked autoencoders are scalable vision learners (2021, FAIR)
    * mask random patches of the input image and reconstruct the missing pixels to an encoder-decoder model, where encoder operate only on the visible patches, and the decoder reconstructs the original image from latent representation and mask tokens 

* SimMIM: a simple framework for masked image modeling (2021, Microsoft)

* iBOT: image BERT pre-training with online tokenizer (2022, ByteDance)
    * propose oneline tokenizer to find a semantic meaningful visual tokenizer for masked image modeling(MIM)

* VPT: visual prompt tuning (2022, Cornell)
    * VPT introduce only a small amount of trainable parameters in the input space while keeping the model backbone frozen


## General Transformer

* Full Stack optimization of transformer infernece: a survey (2023, Berkeley)

### Vision Transformer



## quantization inference 

* Training Quantized Nets: a deeper understanding (2017, Cornell)

* Quantization and Training of neural networks for efficient Integer-Arithmetic-Only(int8) Inference (2017, Google)

* Quantizing deep convolutonal networks for efficient inference: a whitepaper (2018, Google)

* Int4: Low-bit quantization of neural networks for efficient inference (2018, Huawei)

* Q8BERT: Quantized 8Bit BERT (2019, Intel)

* HAWQ: Hessian Aware Quantization of neural networks with mixed-precision (2019, Berkeley)

* Data-Free Quantization: Through weight equalization and bias correction (2019 Qualcomm)

* Int8: Integer Quantization for Deep Learning Inference: Principles and empirical evaluation (2020, NV)

* Trained Quantization thresholds for accurate and efficeint fixed-point(int8) inference of deep neural networks (2020, AMD)

* ZeroQ: novel zero shot quantization framework (2020, Berkeley)

* INT8: Integer Quantization for deep learning inference: principles and empirical evaluation (2020, Nvidia)

* LLM.int8():  8-bit matrix multiplicatin for transformers at Scale (FAIR, 2020)

* Pruning and Quantization for deep neural network accelration: a survey (2021)

* FP8 formats for Deep Learning (2022, NVIDIA)

* FP8 vs INT8 for efficient deep learning inference (Qualcomm, 2023)

* FP8-LM: Training FP8 Large Lanugage Models (2023, Microsoft)

* ZeroQuant-FP: a leap forward in LLm PTQ using floating-point format (Microsoft, 2023)
    * FP8 activation consistently outshines its integer (INT8) equivalent
    
* FP8-Quantization: the power of the exponent (Qualcomm, 2024)
    * for PTQ,  FP8 is better than INT8 in terms of accuracy, and the choice of the numbers of exponent bits is driven by the severity of outliers in the network. 

* SmoothQuant: accurate and efficient post-training quantization for LLMs(2024, MIT)



## Distributed Training and AIOps

* Mixed Precision Training (2018, Baidu & Nvidia)

* Multi-tenant GPU clusters for Deep Leraning Workloads: analysis and implications (2018, Microsoft)

* PipeDream: Generalized Pipeline Parallelism for DNN Training (2019, Microsoft)

* TorchScale: transfomers at scale (2022, Microsoft)




## Multi-Tasks


* multi-task multi-sensor fusion for 3D object detection (Uber 2018)

* Dynamic Task prioritization for multitask learning (2018, Stanford)

* GradNorm: gradietn normalization for adaptive loss balancing in deep multitask networks (2018, Magic Leap)

* Multi-task learning using uncertainty to weight loss for scene geometry and semantics (2018, Cambridge)

* Meta-World: a benchmark and evaluation for multi-task and meta reinforcement learning(2019, Stanford)

* Multi-Task leanring as multi-object optimization (2019, Intel)

* multinet: multi-modal multi-task learning for autonmous driving (2019, Berkeley)

* which tasks should be learned together in multi-task learning (2020, Stanford)

* a survey on multi-task learning (2021)



## General AV Tasks

* Perception, Planning, Control and Coordinatation for autonomous Vehicles(2017, MIT)

* Spatial as Deep: spatial CNN for traffic scene understanding (2017, SenseTime)

* LaneNet: real-time lane detection networks for autonomous driving(Horizon Robotics, 2018)

* AVFI: Fault Injection for Autonomous Vehicle (2019, UIUC)

* overview and empirical analysis of ISP parameters tuning for visual perception in autonomous driving (2019, Valeo)

* FuseMODNet: real-time camera and Lidar based moving object detection for robust low-light autonomous driving (2019, Valeo)

* rethinking CNN frameworks for time-senstive autonomous driving applications: addressing an industrial challenge (2019, UNC)

* A survey of deep leraning techniques for autonomous driving (2020)

* replacing mobile camera ISP with a single deep learning model (2020, ETH)

* Deep mutli-modal object detection and semantic segmentation for autonomous driving: datasets, methods and chanllenges(2020, Bosch)

* Deep Learning Sensor Fusion for Autonomous Vehicles Perception and Localization: a review (2020)

* Scalability in Perception for autonomous Driving: Waymo Open Dataset (2020, Waymo)

* CurveLane-NAS: unifying Lane-Sensitive architecture search and adaptive point blending (2020, Huawei)

* Keep your eyes on the lane: real-time attention-guided lane detection (2020)

* PolyLaneNet: lane estimation via deep polynomial regression (2020)

* Ultra Fast Structure-aware deep lane detection (2020)

* LaneGCN: Learning Lane Graph Representations for motion forecasting (2020, Uber)

* Deep Lerning for image and point cloud fusion in autonomous driving: a review (2021)

* Vision-based vehicle speed estimation: a survey (2021)

* RoadMap: a light-weight semantic map for visual localization towards autonomous driving (2021, Huawei)

* 3TCNN-PP: end-to-end deep learning of lane detection and path prediction for real-time autonomous driving(2021, Tinghua)

* TPNet: trajectory proposal network for motion prediction (2021, SenseTime)

* Prdictive Driver Model(PDM): a technical report (2022, Bosch)
    * PDM for unPlan



## General Segmentation

* FCN: Fully Convolutional Networks for Semantic Segmentation (2016, Berkley)

* DeepLabv3+: Encoder-Decoder with Atrous Separable convolution for semantic image segmentation (2018, Google)

* automated evaluation of semantic segmentation robustness for autonmous driving (2020)

* PointRend: image segmentation as rendering (2020, FAIR)

* SegFormer: simple and efficient design for semantic segmentation with Transformers (2021, NVIDIA)
    * proposed a hierarchically transformer encoders to output multiscale features, then aggregate with MLP 

* MaskFormer: per-pixel classification is not all you need for semantic segmentation (2021, FAIR)
    * a unified mask classification framework for both semantic- and instance-level segmentation, by predicting a set of binary masks, each associated with a single global class label.  

* Mask2Former: Masked-attention mask transformer for universal image segmentation (2022, FAIR)
    * proposed masked attention, to extract localized features by constraining cross-attention within predicted mask regions 

* SAM: segment Anything (2023, FAIR)

* Better Call SAL: towards learning to segment anything in Lidar (2024, NV)

## General Diffusion 

* understanding Diffusion Models: a survey (2022, Google)

* diffusion probabilistic modeling for video generation (2022, California Irvine)

* Diffusion models in vision: a survey (2023)

* ControlNet: adding conditional control to text-to-image diffusion models (2023, Stanford)




## AV BEV/3D Tasks

* stereo vision-based semantic 3D object and ego-motion trakcing for autonomous driving(2018, HKUST)

* Temporal Interplacing Network (2020, Tsinghua)
    * fuse info by interlacing spatial representations from past to future

* 3D object detection for autonomous driving: a survey (2022)

* 3D object detectoin for autonomous driving: a review and new outlooks (2022, Shaoshuai Shi)

* Delving into the Devils of BEV Perception: a review, evaluation and recipe (2022, AI Lab/Li Hongyang)

* 3D object detection from images for autonomous driving: a survey (2023, Wanli Ouyang)

* Cross Modal Transformer(CMT): towards fast and robust 3D object detection (2023, Megvii)
    * image and lidar points tokens to output 3d bbox 

* BEV-LaneDet:  a simple and effective 3D Lane Detection baseline(2023, Haomo)

* Transformer based sensor fusion for autonomous driving: a survey (2023, Motional)

* towards viewpoint robustness in BEV segmentation (2023, NV)

### 3D Mono-View 

* Orthographic Feature Transform for monocular 3D object detection (2018, Cambridge)

* FCOS3D: Fully Convolutional one-stage monocular 3D object detection (2021, CUHK)

* Rope3D: the roadside perception dataset for autonomous driving and monocular 3D object detection task (2022, Baidu)

### Multi-View and fusion solutions 

* BEV-Seg: BEV semantic segmentation using geometry and semantic point cloud (2020, Berkeley)
    * predicts pixel depths and combne with pixel semantics in BEV 

* Lift, Splat, Shoot, encoding images from arbitrary camera rigs by implicitly unprojecting to 3D (2020, Nvidia)

* FIERY:  future instance prediction in BEV from surround monocular cameras (2021 ,Wayve)

* DETR3D: 3D object detection from multi-view images via 3D-to-2D Queries (2021, CMU )

* MVDet: Multiview detection with feature perspective transformation (2021)
    * feature vectors sampled from corresponding pixels in multiple views 

* BEVDet: high performance multi-camera 3D object detection in BEV(2021, PhiGent)
    * image encoder + BEV view transformer + BEV encoder + task-specific head 

* VPFNet: improving 3D object detection with virtual point based Lidar and stereo data fusion (2021, USTC)
    * taking sparse lidar point as multi-modal data aggregation location leading information loss, fixing by aggregate lidar and rgb data at virtual points

* Vision-Centric BEV Perception: a survey (2022, Shanghai AI Lab )

* MatrixVT: efficient multi-camera to BEV transformatoin for 3D perception (2022, Megvii)
    * BEV features as matmul of image features and a sparse feature transporting matrix(FTM)

* M2BEV: multi-camera joint 3D detection and segmentation with unified BEV representation (2022,  NVIDIA)

* DETR4D: direct multi-view 3D object detection with sparse attention (2022, Sensetime)
    * perform cross-frame fusion over past object queries and image features, enabling modeling of temporal information 

* BEVFormer: learning BEV representation from multi-camera images via spatio-temporal transformers (2022, Shanghai AI Lab)
    * in grid-shaped BEV view, aggregate spatial info with cross-attention across camera views, and temporal self-attention to fuse history BEV info 

* BEVDet4D: exploit temporal cues in multi-camera 3D object detection (2022, PhiGent)
    
* BEVDepth: acquisition of reliable depth for multi-view 3D object detection(2022, Megvii)
    * explicit depth supervision utlizing encoded intrinsic and extrinsic parameters + depth correction network

* PETR: position embedding transformation for multi-view 3D object detection(2022, Megvii)
    * encode 3D coordinate info into image features to produce 3D position-aware features 

* PETRv2: a unified framework for 3D perception from multi-camera images(2022, Megvii)
    * the 3D PE achieves temporal alignment on object position of different frames

* Learning Ego 3D representation as Ray Tracing (2022, Huawei)
    * proposed a polarized grid of “imaginary eyes” as the learnable ego 3D representation and formulate the learning process with the adaptive attention mechanism in conjunction with the 3D-to-2D projection.

* semanticBEVFusioN: rethink Lidar-Camera fusion in unified BEV representation for 3D object detection (2022, Bosch)
    * claim the necessity of semantic fusion and propse a semantic-geometric interaction mechanism to maintain modality-specific strengths

* Sparse4D-v1: multi-view 3D object detection with sparse spatial-temporal fusion (2023, Horizon Robotics)

* Sparse4D-v2: recurrent temporal fusion with sparse model (2023, Horizon Robotics)

* StreamPETR: exploring object-centric temporal modeling for efficient multi-view 3D object detection(2023, Megvii)
    * perform online manner and long-term historical information through object queries frame by frame

* Towards better 3D knowledge transfer via masked image modeling for multi-view 3D understanding (2023, MMLab)
    * proposed a Geom enhanced Masked Image Modeling(GeoMiM) to transfer knowledge of Lidar to multi-view camera based 3D detection 

* BEVFusion4D: learning Lidar-Camera Fusion Under BEV via Cross-Modality Guidance and temporal aggregation (2023, SAIC AI Lab)

* CoFil2P: image-to-point cloud registration with coarse-to-fine correspondences for intelligent driving (2023)
    * proposed CoFi-I2P registration network that extracts correspondences in a coarse-to-fine manner to achieve the globally optimal solution

* DiffBEV: conditional diffusion model for BEV perception (2023, PhiGent)
    * exploit diffusion models to generate more comprehensive BEV representations, further a cross-attention module used to fuse the context of BEV feature and semantic content from diffusion model



### 3D Lidar 

* Point Cloud labeling using 3D convolution neural networks (2016)

* PointNet: Deep Learning on Point Sets for 3D Classification and segmentation (2017, Charles Qi)

* VoxelNet: End-to-end learning for point cloud based 3D object detection (2017, Apple)

* Frustum PointNet for 3D object detection from RGB-D data (2018, Charles Qi)

* PointPillars: Fast encoders for object detection from point clouds (2019, Aptiv)

* PointRCNN: 3D object proposal generation and detection from point cloud (2019, UHK)

* End-to-End Multi-view fusion for 3D object detection in Lidar Point Clouds (2019, Waymo)
    * project point clouds in both BEV and perspective view, then fusion

* STD: Sparse-to-Dense 3D object detector for point cloud (2019, Tencent)

* PIXOR: real-time 3D object detection from point clouds (2019, Uber)

* SECOND: sparsely embedded convolutoinal detection (2020, Tusimple) 

* 3DSSD: point-based 3D single stage object detector (2020, UHK)

* an LSTM approach to temporal 3D object detection in Lidar Point Cloud (2020, Google)

* PointPainting: sequential fusion for 3D object detection (2020, Aptiv)
    * projecting lidar points on image based semantic segmentation maps, then fed into lidar encoder

* RandLA-net: efficient semantic segmentation of large-scale point clouds (2020, Oxford)
    * random point sampling with local feature aggregation module to progressively increate the receptive field of each 3D point

* Pseudo-Lidar++: accurate depth for 3D object detection in AD (2020, Cornell)

* PointAcc: efficient Point Cloud Accelerator (2021, MIT)

* Lidar R-CNN: an efficient and universal 3D Object Detector (2021, TuSimple)

* CenterPoint: Center-based 3D object detection and tracking (2021, UT Austin)

* MVP: multimodal virtual point 3D Detection (2021, UT Austin)
    * with 2D detections to generate dense 3D virtual points to augment sparse gt 3D points
    
* LIFT: learning 4D Lidar Image Fusion Transformer for 3D object detection (2021, Alibaba)
    * align 4D input sequential to achive multi-frames multi-modal information aggregation 

* PV-RCNN: Point-Voxel Feature set abstraction for 3D object detection (2021, SenseTime)

* 3D-Man:  3D multi-frame attention network for object detection (2021, Waymo)

* 4D-Net for learned multi-modal alignment (2021, Waymo)

* 3D object detection with PointFormer (2021, Tsinghua)
    * propose local-global transformer to integrate local features with global features.     

* VoxSet: a set-to-set approach to 3D object detection from point clouds (2022, UHK)

* PV-RCNN++: point-voxed feature set abstraction with local vector representation for 3D object detection (2022, Sensetime)
    * improving with proposal-centric sampling and VectorPool for better aggregating local point features 

* CenterFormer: Center-based transformer for 3D object detection (2022, Tusimple)

* A survey of robust Lidar-based 3D object detectoin method in Autonomous Driving (2022)

* FSD v2: improving fully sparse 3D object detection with virtual voxels (2023, TuSimple)



### Depth Estimation and Completion

* dense monocular depth estimation in complex scenes (2016, Intel)
    * produce dense depth map from 2 consecutive frames, with (flow field) motion segmentation module

* MonoDepth-v1: Unsupervised Learning of depth and ego-motion from video (2017, Google)
    * single-view depth net + multi-view pose net 

* MonoDepth-v2: Digging into self-supervised monocular depth estimation (2019, UCL)
    * reprojection loss,  multi-scale sampling,  auto-masking loss 

* a survey on deep learning techniques for stereo-based depth estimation (2020)

* unsupervised monocular depth learning in dynamic scenes (2020, Waymo)
    * jointly training depth estimation, ego-motion and a dense 3D translation field of objects with prior knowledge about 3D translation fields 

* self-supervised monocular depth estimation: solving the dynamic object problem by semantic guidance (2020, Germany)

* removing dynamic objects for static scene reconstruction using light fields (2020)

* learning joint 2D-3D representation for depth completion(2020, Uber)
    * simply stacking 2D features and 3D features

* towards robust monocular depth estimation: mixing datasets for zero-shot cross-dataset transfer (2020, Intel)

* PENet: towards precise and efficient image guided depth completion (2021, Huawei)
    * branch-1: input rgb + sparse depth map to predict dense depth features
    * branch-2: sparse depth map + previous dense depth map to predict dense depth features 
    * fused the two dense depth features by 3D conv

* Revisiting stereo depth estimation from a sequence-to-sequence perspective with transformers(2021, JHU)

* Consistent Depth Estimation in data driven simulation for autonomous driving (2021, MIT)

* MonoRec: semi-supervised dense reconstruction in dynamic environments from a single moving camera(2021)
    * proposed a maskModule to mask moving objects by using photometric inconsistencies encoded in cost volumes

* Insta-DM: learnning monocular depth in dynamic scenes via instance-aware projection consistency(2021, KASIT)
    * propose instance-aware photometric(by off-shelf instance segmentation) and geom consistency loss 

* SC-Depth-v1: unsupervised scale-consistent depth learning from video (2021, TuSimple)
    * proposed a geometry consistency loss to penalize the inconsistency of predicted pepth between adjacent views
    * proposed self-discovered mask to automatically localize & mask out moving objects 

* RigNet: repetitive image guided network for depth completion (2022, NJU)

* SFD: sparse fuse dense towards high quality 3D detection with depth completion (2022, ZJU)
    * propose a 3D ROI fusion strategy to fuse sparse lidar and dense psudeo-lidar from rgb 

* deep depth completion: a survey (2022)

* BEVStereo: enhancing Depth Estimation in multi-view 3D object detection with dynamic temporal stereo (2022, Megvii)
    * propose a temporal stereo method to dynamic scale the matching candidates
    * propose an iterative algorithm to update more valuable candidates for moving candidates

* SC-Depth-v3: robust self-supervised monocular depth estimation for dynamic scenes (2023, TuSimple)
    * introduced pre-trained monocular depth estimator to generate prior pseudo-depth
    * a new loss to boost self-supervised training

* efficient stereo depth estiamtion for Pseudo-Lidar: a self-supervised approach based on multi-input Resnet encoder (2023)

* Metric3D: towards zero-shot metric 3D prediction from a single image (2023, DJi)

* Learning to fuse monocular and multi-view cues for multi-frame depth estimation in dynamic scenes (2023, DJI)
    * fuse multi-view cues (more accurate geometric in static area) and monocular cues(more useful in dynamic areas) with a cross-cue fusion module

* self-supervised monocular depth estimation: let's talk about the weather (2023, UK)
    * pseudo-supervised loss for both depth and pose estimation

* BEVScope: enhancing self-supervised depth estimation leveraging BEV in dynamic scenarios (2023, ZhaoHang's team)
    * self-supervised depth estimation that harnesss BEV features 
    * proposed adaptive loss to mitigate moving objects 
    
* Disentangling object motion and occlusion for unsupervised multi-frame monocular depth (2023, Clemson)
    * proposed dynamic object motion disentanglement module to disentangle object motions 
    * design occlusion-aware cost volume and re-projection loss 

* BEVStereo++: accurate depth estimation in multi-view 3D object detectio nvia dynamic temporal stereo(2023, Megvii)

* CompletionFormer:  depth completion with convolutions and vision transformers (2023, PhiGent)
    * joint conv and attention to construct depth completion in a pyramid structure


### Ego Pose Estimation

* GSNet: joint vehicle pose and shape reconstruction with geometrical and scene-aware supervision (2019, HKU)

* EgoNet: exploring intermediate representation for monocular vehicle pose estimation (2021, SenseTime)
    * propose intermediate geometrical representation for ego-centric orientation 
    * propose a projection invarient loss to incorporate geometry knowledge 

### Tracking & Prediction in 3D 

* exploring the limitations of behavior cloning for autonomous driving (2019)


* CenterTrack: Tracking objects as points (2020, UT Austin)

* 3D Mutli-Object Tracking (AB3DMOT): a baseline and new evaluation metrics (2020, CMU)
    * lidar 3D detector + 3D Kalman filter and Hungarian algorithm to state estimation

* Deep Kinematic models for kinematically feasible vehicle trajectory predictions (2020, Uber)

* PILOT: Efficient Planning by imitation learning and optimisation for safe autonomous driving (2021)

* Scene Transformer: a unified multi-task model for behavior prediction and planning (2021, Google)
    * predicting the behavior of all agents jointly in real-world driving with masking strategy. 

* Binary TCC: A temporal Geofence for Autonomous Navigation (NVIDIA, 2021)
    * time-to-collide as path planning

* mmTransformer: multimodal motion prediction with stacked transformers (2021, UHK)
    * model multimodality at feature level with a set of fixed independent proposals

* Immortal Tracker: tracklet never dies (2021, Tusimple)
    * reveal that pre-mature tracklet termination is the main cause of identity swithces in 3DMOT, 

* SimpleTrack: understanding and rethinking 3D multi-object tracking (2021, TuSimple)

* MOTR: end-to-end multiple-object tracking with Transformer (2022, megvii)
    * introduce  "track query" to model the tracked instances in sequnce of video and propose tracklet-aware label assignment to train track queries and newborn object queries.

* MOTRv2: bootstrapping end-to-end multi-object tracking by pretrained object detectors (2022, megvii)
    
* MUTR3D: a multi-camera tracking framework via 3D-to-2D queries (2022, Tsighua Zhao Hang)
    * introduce 3D  track query to model spatial and apperance coherent track for each objects, wo explicitly rely on spatial and appearance similarity of objects, and map the 3D track query to their 2D image with view transformation 

* PTTR: relational 3D point cloud object tracking with transformer (2022, Nanyang)
    * coarse tracking by matching two sets of point features via cross-attention, 

* Motion Transformer with global intention localization and local movemnet refinement (2023, MPI)
    * proposed MTR that models motion prediction as the joint optmization of global intention localization and local movement refinement, then a prediction refinement module to obtain the final refined prediction

* OmniTracker: unifying object tracking by tracking-with-detection (2023, Fudan)
    * propose the way, where tracking supplyment appearance priors for detection and detection provides tracking with candidate bbox for association

* ViP3D: end-to-end visual trajectory prediction via 3D agent queries (2023, Tsinghua)
    * instead of separate perception and prediction modules, ViP3D use sparse agent queries to detect, track and prediction in one pipeline, where features in spatial and temporal are encoded in agent queries.




### Ground Truth Auto-Labeling System

* Leveraging pre-trained 3D object detection models for fast ground truth generation (2018, Waterloo)

* Efficient Interactive Annotation of segmentatoin dataset with Polygon-RNN (2018, NV)
    * proposed Polygon-RNN to produce polygonal annotations of objects 

* Fast Interactive Object Annotation with curve-GCN (2019, NV)
    * proposed Graph Conv net(GCN) for object annotation with polygons


* LATTE: accelerating Lidar point cloud annotation via sensor fusion, one-click annotation and tracking (2019, Berkeley)

* leveraging temporal data for automatic labeeling of static vehicles (2020, Tornoto)
    * with pretrained 3D detection + multi-frame prediction 
    
* autolabeling 3D objects with differentiable rendering of SDF shape prioris (2020, TRI)
    * proposed a differentiable shape renderer to SDF with normalized object coordinate spaces

* offboard 3D object detection from point cloud sequences (2021, Waymo)
    * proposed multi-frame object detection and object centric refinement model

* auto4D: learning to label 4D objects from sequential point clouds (2021, Uber)
    * tracking online object detection as inital motion path, then refine object size and motion path 

* automatic labelling to generate training data for online Lidar-Base moving object segmentation (2022, German)
    * 1) detect dynamic objects coarsely by occupancy based way, 2) extract segments among the proposals and track trajectories, 3) label moving objects as moving

* MPPNet: multi-frame feature intertwining with proxy points for 3D temporal object detection (2022, MMLab, shaoshuai Shi)
    * per-frame feature encoding, short-clip feature fusion, whole-sequence feature aggregation 

* CTRL: once detected never lost, surpassing human performance in offline Lidar based 3D object detection (TuSimple, 2023)
    * a track-centric offline detector

* DetZero: rethinking 3D object detection with long-term sequential point clouds (2023, Shanghai AI Lab)
    * offline tracker pluse a multi-frame detector to complete the object tracks, then cross-attention object refine module


### Online Mapping

* Predicting semantic map representations from images using Pyramid Occupancy Networks (2020, Cambridge)

* HDMapNet: an online HD map construction and evaluatoin framework (2022, ZhaoHang's team)

* Cross-view transformers for real-time map-view semantic segmentation (2022, UT Austin)
    * learn a mappings from individual camera views into a canonical map-view representation with a camera-aware cross-view attention mechanism.

* Translating images into maps (2022)
    * maping from images or video directly into an BEV map

* MachMap: end-to-end vectorized solution for compact HDMap construction (2023, MachDrive)
    * hdmap construction as the point detection paradigm in BEV space 

* Neural Map Prior for autonomous driving (2023, QiZhi Institute)

* VectorMapNet:  end-to-end vectorized HDMap Learning (2023, Li Auto)
    * predict sparse set of polylines in BEV 

* MV-Map: offboard HDMap generation with multi-view consistency (2023, Fudan)

* MapVR: Online Map Vectorization for autonomous driving: a rasterization perspective (2023)

* MapTR-v1: structured modeling and learning for online vectorized HDMap construction (2023, Horizon Robotics)

* End-to-End vectorized HDMap construction with Piecewise Bezier Curve (2023, Megvii)

* ScalableMap: scalable map learning for online long-range vectorized HDMap construction (2023, U of Wuhan)

* MapEX: Accounting for existing map information when estimating online HDMaps from sensor data (2023, France)

* MapTR-v2: an end-to-end framework for online vectorized HDMap construction (2023, Horizon)

* VMA:  divide-and-conquer Vectorized Map Annotation system for large-scale driving scene (2023)


### Road Segment & Reconstruct

* GndNet: fast ground plane estimation and point cloud segmentation for autonomous vehicle (2020)

* Road Surface reconstruction by Stereo Vision (2020)

* fast ground segmentation for 3D Lidar point cloud based on Jump-Convolution-Process (2021)

* LR-Seg: a ground segmentation method for low-resolution LIdar point clouds (2023, Tsinghua)

* RoME: Towards large scale road surface reconstruction via mesh representation (2023, Horizon Robotics)

* StreetSurf: extending multi-view implicit surface reconstruction to street views (2023, Shanghai AILab)

* PlaNerf: SVD unsupervised 3D plane regularization for NERF large-scale urban scene reconstruction (2023, Huawei)
    * proposed a plane regularization based on singular value decomposition(SVD) and leveraged structural similairty index measure(SSIM) in patch-based loss 

* [mv-map](https://github.com/ZiYang-xie/MV-Map)  2023 

* [ScalableMap](https://github.com/jingy1yu/ScalableMap), 2023 

* [nv 2023 fegr: nerual fields meet explicit geom representatoins for inverse rendering of urban scene](https://research.nvidia.com/labs/toronto-ai/fegr/)

* [google 2022: urban radiance fields](https://urban-radiance-fields.github.io/)


### Occupancy network

* Occupancy Networks: learning 3D reconstruction in fucntion space(2019, MPI)
    *  Occ Networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier.

* Grid-centric traffic scenario perception for autonmous driving: a comprehensive review (2023, Tsinghua)

* FB-OCC: 3D occupancy prediction based on forward-backward view transformation (NV, 2023)
    * based on FB-BEV with joint depth-semantic pretraining,  joint voxel-BEV representation, model scaling up and post-processing

* OCC-BEV: multi-camera unified pre-training 3D scene reconstruction (2023, Peking)
    * BEV features + 3D conv as occupancy decoder combined with prior space occupancy labels from Lidar voxels

* TPVFormer: Tri-Perspective View(TPV) for vision-based 3D semantic occupancy prediction (2023, PhiGent)
    * TPV by BEV with two additional perpendicular planes, so each point in 3D space is represented by summing its projected features on the three planes. proposed a TPV encoder to fuse TPV features 

* OccFormer: dual-path transformer for vision-based 3D semantic occupancy prediction (2023, PhiGent)
    * decomposing 3D processing into local and global transformer pathways along the horizontal plane, and Mask2Former is used for 3D semantic occupancy.

* SurroundOcc: multi-camera 3D occupancy prediction for autonomous driving(2023, PhiGent)
    * multi-view cameras -> 2D-3D attention -> 3D volume space -> 3D conv -> dense occ prediction pipeline by fusing multi-frame Lidar scans and fill holes with Poisson Recon

* UniOCC: unifying vision-centric 3D occupancy prediction with geometric and semantic rendering (2023, Xiaomi Car)
    * mutli-view cameras -> 2Dto3D view transfomer -> 3D voxel features -> 2 branches: 1. geometry mlp + 2. semantic mlp -> occupancy prediction. while geometry and semantic features are supervised by NERF volume rendering 

* VoxFormer: sparse voxel transformer for camera-based 3D semantic scene completion(2023, NV)
    * initial with sparse set of voxel queries from depth estimation, followed by masked autoencoder to propgaget the info to all the voxles by self-attention. 




## AV Test & Simulations 

* On a Formal model of safe and scalable self driving cars(Mobileye, 2017)
    * Responsibility sensitive safety(RSS)

* TrafficNet: an open naturalistic Driving Scenario Library (2017, UMich)

* DeepTest: automated testing of deep nerual network driven autonomous cars (2018)

* Driving Simulation Technologies for sensor simulation in SiL and HiL environments (2018, dSpace)

* DeepRoad: GAN based metamorphic testing and input validation framework for ADS (2018, UTexas at Austin)

* Test your self-driving algorithm: an overview of publicly available driving datasets and virtual test enviroments (2019)

* Towards corner case detection for autonomous driving (2019, Volkswagen)

* Scalable end-to-end autonomous vehicle testing via rare-event simulation(2019, MIT)
    * important-sampling to accelerate rare-event probability evluation, by estimating the probability of accident under a base distribution governing standard traffic behavior

* Failure-scenario maker for rule-based agent using multi-agent adversarial reinforcement learning and its application to autonomous driving (2019, IBM)

* Generating Adversarial Driving Scenarios in high-fidelity simulation (Toronto, 2019)
    * Bayesian to generate poorly behaviors to increase possibility of collision with virtual pedestrains and vehicles

* Generation of scenes in intersections for the validatoin of hihgly automated driving functions(2019, Bosch)

* ML based fault injection for autonmous driving (2019, NV)

* Multimodal safety-critical scenarios generations for decision-maker algorithms evaluation(2020, CMU)

* Neural Bridge Sampling for evluating safety-critical autonomous system(2020, Stanford)
    * a rare-event simulator to find failure modes and estimate their rate of occurrence 

* Cam2BEV: a sim2real DL approach for the transformation of images from multiple vehicle-mounted camreas to a semantically segmented image in BEV (2020)

* SimNet: learning reactive self-driving simulations from real-world observations(2021, Lyft)
    * Markov Process + DL to model state distribution and transition functions

* AdvSim: Generating safety-critical scenarios for self-driving (2021, Uber)
    * adversarial framework to generate scenarios for lidar based system

* DriveGAN: towards a controllable high-quality neural simulation (2021, NV)

* SceneGen: learning to generate realistic traffic scenes(2021, Uber)
    * with ego state + hdmap to generate scenarios

* Enhancing SUMO simulator for simulation based testing and validating of autonomous vehicles(2021, UMich)
    * sumo + openAI Gym

* efficient and effective generation of test cases for pedestrain detection - search based software testing of Baidu Apollo in SVL (2021)

* Real Time Monocular vehicle velocity estimation using synthetic data (2021, Oxford)

* Imaging the road ahead:  multi-agent trajectory prediction via differentiable simulation (2021)
    * build a fully differentiable simulator for multi-agent trajectory prediction

* VISTA 2.0 : an open, data-driven simulator for multimodal sensing and policy leanring for autonomous vehicles (2021, MIT/Han)
    * 

* IterSim: interactive traffic simulation via explicit relation modeling(2022, ZhaoHang's Team)
    * input as ego trajectory, InterSim inference agents trajectories

* scenario Diffusion: controlable driving scenario generation with diffusion (2023, Zoox)
    * combined latent diffusion, object detectiion and trajectory regression to generate distributions of synthetic agetn pose, additionaly control with condition on map and set of tokens describing desried scenario

## SDG 

* A lidar point cloud generator: from a virtual world to autonomous driving (2018, Berkeley)

* Synthetic data for deep learning: a survey (2019, Synthesis.AI)

* Meta-Sim: learning to generate synthetic datasets (2019, NVIDIA)
    * a generative model learns to modify attributes of scene graphs

* Meta-Sim2: unsupervised learning of scene structure for synthetic data generation (2020, NVIDIA)
    * RL to learn sequentially sample rule from a given probabilistic scene grammer

* Synthetic data generation using imitation training (2020, NV)

* SurfelGAN: synthesizing realistic sensor data for autonomous driving (2020, Waymo)


* SDG: towards optimal strategies for training self-driving perception models in simulation (2021, NVIDIA)

* Understanding Domain Randomization for sim-to-real transfer (2021, Peking)

## Nerf & Scene Reconstruct

* Learning Category-specific Mesh reconstruction from image collections (2018, Berkeley)
    * the shape is represented a deformable 3D mesh model of an object category where a shape is parameterized by a learned mean shape and per-instance predicted deformation.

* DeepSDF: learning continuous signed distance functions for shape representation (2019, Facebook)
    * representation implicitly encode a shape's boundary as zero-level-set of the SDF while explicitly representing the classification of space as interior or outer

* NERF: representing scenes as neural radiance fields for view synthesis (2020, Berkeley)

* Points2Surf:  learning implicit surfaces from point clouds (2020)
    * learning a prior by a combination of detail local patches and coarse global info to imporove reconstruct performance and accuracy

* NeRV: neural reflectance and visibility fields for relighting and view synthesis (Google, 2020)
    * proposed a model to represent scene as a continous volumetric function, parameterized as MLPs, with 3D locations as input, and output scene properties(volumen density, surface normal, material parameters, visibility) at the input location

* Free view synthesis (2020, Intel)
    * calibrate input images via SFM, and create a coarse scaffold via MVS, which further used to create a proxy depth map for novel view, then a recurrent encoder-decoder network process the proxy depth to joint features from nearby view and output novel view.

* RegNef: regularizing neural radiance fields for view synthesis from sparse input(2021, Google)
    * observed majority of artifacts in sparse input scenarios are caused by errors in the estimated scene geometry. propose solution by regularizing the geometry and appearance of patches rendered from unobserved viewpoints, and annealing(退火) the ray sampling space during training. 

* dense depth priors for neural radiance fields from sparse input views (Google, 2021)
    * first utilize sparse points from SFM to depth completion, then use these depth estimation as constraints for nerf training     

* Neural-Pull: learning SDF from point clouds by leanring to pull space onto surfaces( 2021)
    * train a network to pull query 3D location to their closest points on the surface by using predicted SDF values and the gradient at query locations

* Ref-Nef: structured view-dependent appearance for neural radiance fields (2021, Google)
    * observed nerf failed to accurately capture and reproduce the appearance of glossy surfaces, proposed solution to replace view-dependent outgoing radiance parameters with a representation of reflected radiance.

* Nerf in the wild: neural radiance fields for unconstrained photo collections (2021, Google)

* Stable View Synthesis (2021, Intel)
    * get geometric scaffold vai SFM, each point on this 3D scaffold asociated with view rays and feature vectors that encode the appearance of this point in images. 

* NKF: Neural Fields as Learnable kernels for 3D reconstruction (2021, NV)
    * kernel methods with appropriate inductive bias are extremly effective for reconstructing shapes. 

* Neural RGB-D surface reconstruction (2021, Google)
    * instead of volum representation of surface, here propose surface representation wihth truncated SDF, and integrated into NERF. 

* Mending neural implicit modeling for 3D vehicle reconstruction in the wild (2021, Uber)
    * nerf with prior shape latent-code, test-time regularized optimization, a deep discriminator as shape prior and a learning strategy to learn shape priors on synthetic data

* Volume Rendering of neural implicit surfaces (2021, FAIR)
    * improve geom representation and reconstruct in volume rendering, by modeling volume density as a Laplace's cumulative distribution func applied to SDF representation 

* MonoSDF: exploring monocular geometric cues for neural implicit surface reconstruction (2022) 
    * demonstrate that depth and normal cues, predicted by general-purpose monocular estimators, significantly improve reconstruct quality and optimization time.

* Shape, Light and material decomposition from images using Monte Carlo rendering and denoising (2022, NV)
    * proposed a realistic shading model, incorporating ray tracing and Monte Carlo integration to substantially improves decomposition into shape, materials and lighting. to address noise during Monte Carlo integration, further with multi importance sampling and denoising during rendering pipeline

* Mip-NERF360: unbounded anti-aliased neural radiance fields (2022, Google)
    * use a non-linear scene parameterization and distortion-based regularizer for unbounded scenes

* NVDiffrec: extracing triangular 3D models, materials and lighting from images (2022, NV)
    * leverage differentiable rendering to disentangle 3D mesh with spatial-varying material and environment lighting

* Control-Nerf: editable feature volumes for scene rendering and manipulation (2022, )
    * proposed a model to decouple volume rendering from scene-specific geometry and appearance.

* Noise2NoiseMapping: learning SDF from noisy 3D point clouds via noise to noise mapping(2023, )
    * proposed a loss which enable statistical reasoning on point clouds and maintain geometric consistency

* NKSR: Neural Kernel Surface Reconstruction (2023, NV)
    * recovering a 3D surface from an input point cloud, robost in large scale and noise

* Neuralangelo: high-fidelity neural surface reconstruction (2023, NV)  
    * numerical gradients for higher-order derivatives as smoothing operator and coarse-to-fine optimization on hash grids to control level of details

* F2-nerf: fast neural radiance field training with free camear trajectories (2023, MPI)    
    * proposed a space-warning method to handle arbitrary trajectoreis in the grid-base nerf framework

* LightSim: neural lighting simulation for urban scenes (2023, Wabbi)
    * LightSim automatically builds lighting-aware digital twins at scale from collected raw sensor data and decomposes the scene into dynamic actors and static background with accurate geometry, appearance, and estimated scene ligh. 

* NeuRAD: neural rendering for autonmous driving (2023, Zenseact)
    * nerf scene representation with neural camera, lidar models 

* UniSim: a neural closed-loop sensor simulation (2023, Waabi)

* Car-Studio: learning car radiance fields from single view and endless in-the-wild images (2023)

* NerfXL: Nerf at any scale with multi-GPU (2024, NV)

* NeRFect Match: exploring NERF features for visual localization (2024, NV)



### Acclerated Nerf 

* Plenoxels: radiance fields without neural networks (2021, Berkeley)
    * represent scene as a sparse 3D grid with spherical harmonics

* PlenOctrees: for real-time rendering of neural radiance fields (2021, Berkeley)
    * pre-tabulating nerf into PlenOctree

* InstantNPG: instant neural graphics primitives with a multiresolution hash encoding (2021, NV)
    * replace MLP nerf with a smaller neural network with multi-resolution hash table of trainble vectors 

* DirectVoxGO: direct voxel grid optimization, super-fast convergence for radiance fields reconstruction (2022, ETHU)
    * adopt a scene representation as a density volx grid for scene geometry and a feature voxel grid for scene geometry and a feature voxel grid with a shallow network for complex view-dependent appearance.

* Improved Direct Voxel Grid optimization for radiance fields reconstruction (2022, ETHU)

* TensoRF: tensorial radiance fields (2022, ShangHai Tech)
    * model radiance fields as 4D tensor, then factorize 4D tensor into multi compact low-rank tensor components

### large scale Nerf 

* Urban Radiance Fields (Google, 2021)
    * extend nerf by leveraging lidar supervision to address exposure variations between captured images; and leveraging pretrained segment models to supervise densities on rays pointing at sky. 

* Mega-Nerf: scalable construction of large-scale nerfs for virtual fly-throughs (2021, CMU)

* Block-Nerf: scalable large scene neural view synthesis (Waymo, 2022)
    * demonstrate when rendering nerf in city scale, requires decompose into smaller neural volumes, with additional appearance embeddings to align appearance between adjacent nerf blocks.

* FEGR: Neural Fields meet explicit geometric representations for inverse rendering of urban scenes (2022, NV)
    * nerf methods achieve impressive fidelity of 3D reconstruction, while bake lighting and shadows into the radiance field, here proposed an inverse rendering framework for large urban scenes capable of jointly reconstructing the scene geometry, spatially-varying materials and HDR lighting from RGBs with optional depth.

* Neural Light Field Estimation for street scenes with differentiable virtual object insertion (2022, NV)

* READ: large-scale neural scene rendering for autonmous driving (2022, Alibaba)


### Multi-view 

* Multiview neural surface reconstruction by disentangling geometry and appearance (2020)

* UNISURF: unifying neural implicit surfaces and radiance fields for multi-view reconstruction (2021, MPI)  
    * surface models and radiance fields(nerf) can be formulated in a unified way, enabling both surface and volume rendering using the same model

* NeuralWarp: improving neural implicit surfaces geometry with patch warps (2022)
    * prpose to add photo-consistency term acrros multi-views (by measuring similarity with predicted occupancy and normals of 3D points alone each ray) to standard neural rendering

* NeuS: learning neural implicit surfaces by Volmen Rendering for multi-view reconstruction (2023, MPI)
    *  reconst objects from 2D images, by representing surface as zero-level set of SDF and developing a new volume rendering method to training a neural SDF representation

* Neus2: fast learning of neural implicit surfaces for multi-view reconstruction (2023, MPI)
    * parameterized neural surface representatoin by multi-resolution has encodings 


### Dynamic Scene

* Object centric neural scene rendering (2020, Stanford)
    * propose a representation that modles per-object light transport implicitly using a lighting and view-dependent neural network. 

* Neural Scene Graphs for dynamic scenes (2021, Princeton)
    * decompose dynamic scenes into scene graphs, and proposed a learned scene graph representation, which encodes object transformation and radiance

* D-nerf:  neural radiance fields for dynamic scenes (2021, MPI)    
    * consider time as an additional input, split the learning process into 2 stages: first encode the scene in a canonical space, then map this canonical representation into the deformed scene at a particular time.

* D2Nerf:  self-supervised decoupling of dynamic and static objects from a monocular video(2022, Cambridge)
    * represent the dynamic objects and static background by 2 separate neural radiance fields, proposed a new los to correct separation, and a shadow network to decouple dynamically moving shadows.

* Panoptic neural field: a semantic object-aware neural scene representation (2022, Google)
    * decompose a scene into a set of objects and background, each object represented by an oriented 3D bbox and MLP, the background is represented by a MLP with additinoaly semantic labels output. 

* MARS: an instance-aware, modular and realistic simulator for autonomous driving (2023, Tqinghua/Zhao Hao)

* EmerNerf: emergent spatial-temporal scene decomposition via self-supervision (2023, NV)
    * learning spatial-temporal representation of dynamic driving scenes, by first stratifies scenes into static and dynamic fields, purly from self-supervision; then learn a flow field from the dynamic field, which used to aggregate multi-frames features to dynamic rendering. 

### Semantic Nerf 

* Semantic-NerF: in-place scene labelling and understanding with implicit scene understanding (2021, London)
    * extend NERF to jointly encode semantics with appearance and geometry, so that complete and accurate 2D semantic labels achieved

* Sem2NerF: converting single-view semantic masks to neural radiance field (2022, NTU)
    * encoding semantic mask into latent code taht controls 3D scene representation with region-aware learning strategy

* Panoptic Nerf: 3D-to-2D label transfer for panoptic urban scene segmentation (2022, ZJU)

* FeatureNerf:  learning generalizable NERFs by distilling foundation models (2023, Tusimple)
    * leverage 2D pre-trained foundation models(DINO, Latent Diffusion) to 3D space via neural rendering, a.k.a map 2D images to continuous 3D semantic feature volumes.

* Segment Anything in 3D with Nerf (2023, Huawei)

### Lidar Nerf

* LidarSim: realistic Lidar simulation by leveraging the real world (2020, Uber)
    * a lidar sim model based on ray casting in virtual 3D traffic scenes.

* NLF: Neural Lidar fields for novel view synthesis (2023, NV)
    * combine neural rendering with detailed, physically motivated Lidar sensing process 

* DyNFL: dynamic Lidar resimulation using compositional neural fields(2023, NV)
    * prposed nerual based simulation of Lidar in dynamic scenes. 

* GPU rasterization based 3D Lidar simulation for deep learning (2023, MPI)

* Nerf-Lidar: generating Realistic Lidar point clouds with neural radiance Fields (2023, Fudan)
    * reconst the scene with nerf, then util a virtual lidar simulator to generate lidar points(), then create point-wise semantic label. supervision with depth, rgb, lidar and semantic.


### Nerf for SDG

* PreSIL: precise synthetic image and Lidar dataset for autonomous vehicle perception(2019, Waterloo)
    * SDG based on GTA gamer

* GeomSim:  realistic video simulation via geometry-aware composition for self-driving (2021, Uber)
    * propose a geometry-aware image composition process, in which 1) learn geometry and appearance from real sensor data, 2) propose an realistic object placements in a given scene, 3) render the scene with dynamic object from 1), 4) compose and blend the rendered image segments.

* CalraScenes: a synthetic dataset for odometry in autonomous driving (2021)

* is synthetic data from generative model ready for image recognition? (2023, UHK)

* NeuSim: Reconstructing objects in the wild for realistic sensor simulation (2023, Waabi)
    * represent the object surface as neural SDF and leverage both Lidar and camera to reconstruct smooth and accurate geometry and normals, and model object appearance with physics inspired reflectance representation.

* SHIFT: a synthetic driving dataset for continuous multi-task domain adaptation (2021, MPI)


### GAN in Nerf

* styleGAN-v1: a style-based generator architecture for generative adversarial network (2019, NV)

* styleGAN-v3:  alias-free generative adversarial networks (2021, NV)

* GRAF: generative radiance fields for 3D-aware image synthesis (2021, MPI)

* GIRAGGE: representation Scenes as compositional generative Neural Fetaures Fields (2021, MPI)

* efficient geometry-aware 3D generative adversarial network (2022, NV)

* A comprehensive survey of AIGC: a history of generative AI from GAN to ChatGPT (2023, Lehigh)




## 3D Gausssian

* 3D Gaussian splatting for real-time radiance fields (2023, MPI)

* DrivingGaussian: Composite Gaussian splatting for surrounding dynamic autonomous driving scenes (2023, Peking)

* Street Gaussian for modeling dynamic urban scenes (2024, Li Auto)



## General LLMs & VLMs

* PPO: proximal policy optimization algorithms (2017, OpenAI)

* parameter efficient transfer leraning for NLP (2019, Google)

* P-tuning: GPT understands too (2021, Tsinghua)

* the power of scale for parameter-efficient prompt tuning (2021, Google)

* VisualGPT: data-efficient adaptation of pretrained language models for image caption (2021, KAUST)

* ViLT: visiona-and-lanuage transformer without convolution or region supervision (2021)

* CLIP: learning transferable visual models from natural language supervision (2021, OpenAI)

* FlashAttention: fast and memory efficient exact attention with IO-awarenewss (2022, Stanford)

* CoOp:  learning to prompt for vision-lanuage models (2022, Nanyang)

* BLIP: bootstrapping lanuage-image pret-training for unified vision-lanuage understanding and generation (2022, Salesforce)

* Hierarchical text-conditional image generation with CLIP latents (2022, OpenAI)
    * first generate a prior by CLIP, then use a diffusion model to generate image 

* Flamingo: a visual lanuage model for few shot learning (2022, DeepMind)

* VL-Adapater: parameter-efficient transfer learning for vision-and-language tasks(2022, UNC)

* Vision Language pre-training: basics, recent advances and future trends (2022, Micorsoft)

* LLama: open and efficient foundation lanuage models (2023, Meta)

* self-instruct: aligning lanuage models with self-generated instructions (2023, Washington)

* SmoothQuant: accurate and efficient post-training quantization for large lanuage models (2023, MIT/Han)

* multimodal foundation models: from specialist to general-purpose assistants (2023, Microsoft)

* towards a unified agent with foundation models (2023, DeepMind)

* the false promise of imitating proprietary LLMs (2023 Berkely)

* Chain of thought prompting elicits reasoning in llm (2023, Google)

* RT2: vision-language-action models transfer web knowledge to robotic control (2023, Google)

* Llama-adapter v2: parameter-efficient visual instruction model (2023, SHAI Lab)

* LLM-adapters: an adapter family for parameter-efficient fine-tuning on LLM (2023, Singapore)

* Qwen-VL: a versatile VLM for understanding, localization, text reading and beyond (2023, Alibaba)

* LLAVA: visual instruction tuning (2023, Microsoft)

* MiniGPT4: enhancing vision lanuage understanding with advacned LLM (2023, KAUST)

* a systematic survey of prompt engineering on vision-lanuage foundation models (2023, Oxford)

* a survey of resource-efficient LLM and multimodal foundation models (2024)

* efficient LLMs: a survey (2024)

* a survey on Hallucination in large vision language models (2024, Huawei)

* vision language models for vision taks: a survey (2024, Nanyang)

* vision language navigation with embodied intelligence: a survey (2024)




## Drive VLM 

* Driving with LLMs: fusing object-level vector modality for explainable autonomous driving (2023, Wayve)

* Drive Like a human: rethinking autonmous driving with LLM (2023, SHAI Lab)

* DiLU: a knowledge-driven approach to autonomous driving with LLM (2023, SHAI Lab)

* LMDrive: closed-loop end-to-end driving with LLM (2023, MMLab)

* DriveLM: Drivign with graph visual question answering (2023, SHAI Lab)

* ADAPT: action-aware driving caption transformer (2023, Tsinghua)

* on the road with GPT-4Vision: early exploratoins of VLM on autonmous driving (2023, SHAI Lab)

* DriveGPT4: interpretable end-to-end autonomous driving via LLM (2024, UHK)

* BEV-CLIP: multi-modal BEV retrieval methodology for complex scene in autonmous driving (2024, Li Auto)

* DriveVLM: the convergence of autonmous driving and large vision-lanuage models (2024, Li Auto)




## Generative 3D 

* Get3D: a generative model of hihg quality 3D textured shapes learned from images (2022, NV)

* Magic3D: high-resolution text-to-3D content creation (2022, NV)


## accleration computing 

*  Roofline: an insightful visual performance model for floating-point programs and multicore arch (2018)





## reference 

1. [conenction to diffusion models and others](https://yang-song.net/blog/2021/score/)
2. [Diffusion Model 中的条件正态分布计算](https://zhuanlan.zhihu.com/p/604912763)
3. [吴海波：Diffusion Model 导读](https://zhuanlan.zhihu.com/p/591720296)
4. [原理+代码：Diffusion Model 直观理解](https://zhuanlan.zhihu.com/p/572161541)
5. [Occupancy Network综述](https://zhuanlan.zhihu.com/p/611625314)
6. [BEV纯视觉感知算法笔记](https://zhuanlan.zhihu.com/p/633624413)
7. [BEV感知学习](https://www.zhihu.com/column/c_1637492524494348288)
8. [openDriveLan AD23 Challenge](https://opendrivelab.com/AD23Challenge.html#Track3)
