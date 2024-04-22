
## 点云感知算法

* [入门点云3d目标检测](https://www.zhihu.com/question/448472033/answer/2136340327)
* [igear: 万字综述： 基于lidar点云的3d目标检测深度学习方法](https://zhuanlan.zhihu.com/p/532803205)

* [激光雷达：3D物体检测算法](https://zhuanlan.zhihu.com/p/390156904)
    * 萌芽期: 
        * 把点云映射到正视图，VeloFCN
        * 把3D点云映射到俯视图，MV3D，将3D点云同时映射到正视图和俯视图，并与2D图像数据进行融合
    
    * 起步期:
        * VoxelNet，将点云量化为网格数据
            * Voxel Partition -> Grouping -> Random Sampling -> Stacked Voxel Feature Encoding -> Sparse 4D Tensor 
        * PointNet++， 处理非结构化的数据点
    
    * 发展阶段:
        * VoxelNet based optimization
            * SECOND, sparse conv 
            * PointPillar, 
        * Point based optimization 
            * Point-RCNN
            * 3D-SSD 
                * FP（Feature Propagation）层和细化层（Refinement）是系统运行速度的瓶颈。
        
        * Voxel + Point fusion
            * 来回顾一下Voxel和Point的主要问题。前者非常依赖于量化的参数：网格大的话信息损失比较大，网格小的话的计算量和内存使用量又非常高。后者很难提取邻域的上下文特征，并且内存的访问是不规则的（大约80%的运行时间都耗费在数据构建，而不是真正的特征提取上）

        * 落地期 2020+ ：
            * CenterPoint
                * 在第一阶段中，其特征提取的主干网络可以采用VoxelNet或者PointPillar的方式，得到的是一个HxWxC的二维图像加通道的数据结构.检测网络的部分和CenterNet非常相似，只是针对BEV数据非常稀疏的特点，调整高斯分布的参数，以增加正样本的数量
                * 在第二阶段中，从已经预测的BBox出发，收集BBox边缘的点特征，用一个MLP对置信度和BBox参数进行细化

    * VoxelNet/SECOND/PointPillar/CenterPoint等Voxel-Based方法
    * PointNet/PointNet++/Point-RCNN等Point-Based方法


* pointNet(2017)
    * symmetry function for unordered input
        * approximate a general function defined on a point set by applying a symmetric function on transformed elements in the set
            1. multi mlp on each point 
            2. max pooling layer to aggregate point-wise feature, output global point cloud feature `f_i` 
    * for classification task:
        * assign mlp classifier on the global features 
    * for segmentation task: 
        * local & global information aggregation
            * concatenating the global feature with each of the point features. 
            * Then we extract new per point features based on the combined point features - this time the per point feature is aware of both the local and global information.
    * joint alignment network
        * the learnt representation by our point set is invariant to these transformations 
        * We predict an affine transformation matrix by a mini-network and directly apply this transformation to the coordinates of input points
        * We can insert another alignment network on point features and predict a feature transformation matrix to align features from different input point clouds.
    

    * [zhihu](https://zhuanlan.zhihu.com/p/264627148)
    * [code]

* pointNet++(2017)
    * partition the set of points into overlapping local regions by the distance metric of the underlying space.
    * Our hierarchical structure is composed by a number of set abstraction levels. At each level, a set of points is processed and abstracted to produce a new set with fewer elements. The set abstraction level is made of three key layers: Sampling layer, Grouping layer and PointNet layer.
    * Set Abstraction Layer(SA)
        * sampling layer :
            *  use iterative farthest point sampling (FPS) to choose a subset of points
        * grouping layer:
            * The input is a point set of size N * (d + C) and the coordinates of a set of centroids of size N0 * d. The output are groups of point sets of size N' * K * (d + C) 
            * K is the number of points in the neighborhood of centroid points
            * K varies across groups but the succeeding PointNet layer is able to convert flexible number of points into a fixed length local region feature vector           
            * Compared with kNN, ball query’s local neighborhood guarantees a fixed region scale thus making local region feature more generalizable across space
        * pointNet layer
            * input:  N' * K * (d + C)
            * output is abstracted by its centroid and local feaure  N' * (d + C')
    * Feature learning under non-uniform sampling density 
        * density adaptive PointNet layers (Fig. 3) that learn to combine features from regions of different scales when the input sampling density changes
        * In PointNet++, each abstraction level extracts multiple scales of local patterns and combine them intelligently according to local point densities
        * multi-scale grouping 
            * to apply grouping layers with different scales followed by according PointNets to extract features of each scale. Features at different scales are concatenated to form a multi-scale feature.
            * train the network to learn an optimized strategy to combine the multi-scale features
        * multi-resolution grouping
            * When the density of a local region is low, the first vector may be less reliable than the second vector,
            * when the density of a local region is high, the first vector provides information of finer details
    * point feature propagation for set segmentation
        * adopt a hierarchical propagation strategy with distance based interpolation and across level skip links
        1. achieve feature propagation by interpolating feature values f of Nl points at coordinates of the N(l-1) points
        2. then concatenated with skip linked point features from the set N(l-1) abstraction level
        3. Then the concatenated features are passed through a “unit pointnet”
        4. A few shared fully connected and ReLU layers are applied to update each point’s feature vector
        5. The process is repeated until we have propagated features to the original set of points

    * [zhihu](https://zhuanlan.zhihu.com/p/266324173)


* VoxelNet(2017)
    * we close the gap between point set feature learning and RPN for 3D detection task.
    * VoxelNet divides the point cloud into equally spaced 3D voxels, encodes each voxel via stacked VFE(voxel feature encoding) layers, and then 3D convolution further aggregates local voxel features, transforming the point cloud into a high-dimensional volumetric representation. Finally, a RPN consumes the volumetric representation and yields the detection result.
    * feature learning network
        * voxel partition
        * grouping: group the points according to the voxel they reside in. 
        * random sampling
            * we randomly sample a fixed number, T, of points from those voxels containing more than T points. 
            * to decreases the imbalance of points between the voxels
        * stacking voxel feature encoding(VFE)
            1. compute the local mean as the centroid of all the points in the non-empty voxel, e.g. (vx, vy, vz)
            2. augment each point `pi` with the relative offset, e.g. `p^` [xi, yi, zi, ri, xi-vx, yi-vy, zi-vz]
            3. `p^` -> FCN(fc + BN + ReLU) to get point features
            4. max pooling accross all point features `f` to get locally aggragated feature `f^`
            5. aggregate each `f` with `f^` to form the point-wise concatenated feature `f^out`
            6. The voxel-wise feature is obtained by transforming the output `f^out` of VFE-n via FCN and applying element-wise Maxpool 
        * Because the output feature combines both point-wise features and locally aggregated feature, stacking VFE layers encodes point interactions within a voxel and enables the final feature representation to learn descriptive shape information.
        * sparse tensor representation
            * The obtained list of voxel-wise features can be represented as a sparse 4D tensor, of size C  D0  H0  W'
    * Conv Middle layers
        * applies 3D conv, BN layer nad ReLU layer sequentially, to aggregate voxel-wise features within a progressively expanding receptive field,adding more context to the shape description.
    * Region Proposal Network
        * input: feature map from conv middle layer 

    * TODO: 点云哈希查询s
    * [zhihu](https://zhuanlan.zhihu.com/p/40051716)



* SECOND: sparsely embedded conv detection  (Oct 2018 )
    * Spatially sparse convolutional networks are introduced for LiDAR-based detection and are used to extract information from the z-axis before the 3D data are downsampled to something akin to 2D image data.
    * network arch:
        * point cloud grouping
            * For the detection of cars and other objects in related classes, we crop the point cloud based on the ground-truth distribution at [􀀀3, 1]  [􀀀40, 40]  [0, 70.4] m along the z  y  x axes
        * voxelwise feature extractor
            * voxelNet takes all points in the same voxel as input  
            * use a fully connected network (FCN) (linear layer + bn + ReLu) to extract point-wise features 
            * Then, it uses elementwise max pooling to obtain the locally aggregated features for each voxel.
            * Finally, it tiles the obtained features and concatenates these tiled features and the pointwise features together
        * sparse conv middle layer 
            * sparseConv: output points are not computed if there is no related input point
            * submanifold convolution [27] restricts an output location to be active if and only if the corresponding input location is active.
            * TODO: a detail sparse conv algorithm 
            * TODO: rule generation algorithm
        * sparse conv middle extractor 
            * used to learn information about the z-axis and convert the sparse 3D data into a 2D BEV image
            0. consists of two phases of sparse convolution
            1. Each phase contains several submanifold convolutional layers and one normal sparse convolution to perform downsampling in the z-axis.
            2. After the z-dimensionality has been downsampled to one or two, the sparse data are converted into dense feature maps.
        * RPN 

    * [zhihu](https://zhuanlan.zhihu.com/p/356892010)
        * 考虑到VoxelNet通过Feature Learning Network后获得了稀疏的四维张量，而采用3D卷积直接对这四维的张量做卷积运算的话，确实耗费运算资源。SECOND作为VoxelNet的升级版，用稀疏3D卷积替换了普通3D卷积


* PointPillars (2019 nuTonomy)
    * 3D lidar point perception background: 1) direct 3D ,  2) projection to 2D BEV then with 2D CNN 
    * PointPillars accepts point clouds as input and estimates oriented 3D boxes for cars, pedestrians and cyclists.
    * step1: feature encoder from point clouds to a sparse pseudo-image 
        * the point cloud is discretized into an evenly spaced grid in the x-y plane, creating a set of pillars `P`
        * The points in each pillar are then augmented with the arithmetic mean of all points in the pillar
        * encoded with simplified PointNet and scatter features back to original pillar locations to create the pseudo-image         
    * step2: 2d cnn to get features 
    * step3: detection head(ssd) to detects and regress 3d box
        * instead given a 2D match, the height and elevation become additional regression targets.

    * [github:]()




* PointRCNN (2019)
    * stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in  the canonical coordinates to obtain the final detection results.
    * bottom-up 3d proposal generation via point cloud segmentation 
        * All 3D objects’ segmentation masks could be directly obtained by their 3D bounding box annotations, points inside 3d box considered as foreground points
        * we learn point-wise features to segment the raw point cloud and to generate 3D proposals from the segmented foreground points simultaneously
    * learning point cloud representation
    * foreground point segmentation 
        * Given the point-wise features encoded by the backbone point cloud network, we append one segmentation head for estimating the foreground mask and one box regression head for generating 3D proposals.
    * bin-based 3d bounding box generation 
        * We observe that using bin-based classification with cross-entropy loss for the X and Z axes instead of direct regression with smooth L1 loss results in more accurate and robust center localization.
        * The targets of orientation  and size (h;w; l) estimation are similar to `frustum` 2017, divide the orientation 2 into n bins, and calculate the bin classification target bin(p)  and residual regression target res(p)  in the same way as x or z prediction.
        * after NMS we keep top 300 proposals for training the stage-2 sub-network.
    * point cloud region pooling 
        * refining the box locations and orientations based on the previously generated box proposals
        * For each 3D box proposal, bi = (xi; yi; zi; hi;wi; li; i), we slightly enlarge it to create a new 3D box bei= (xi; yi; zi; hi + ;wi + ; li + ; i) to encode the additional information from its context. 

    * canonical 3d bbox refinement
        * canonical transformation
        * feature learning for box proposal refinement 
            * For each proposal, its associated points’ local spatial features ~p and the extra features [r(p);m(p); d(p)] are first concatenated and fed to several fully-connected layers to encode their local features to the same dimension of the global features f (p). Then the local features and global features are concatenated and fed into a network following the structure  of [28] to obtain a discriminative feature vector for the following confidence classification and box refinement
                * 缺少深度信息的一个操作：local localization feature + global localization -> 

    * 总结：思路清晰，但感觉很重。 


* MVF: e2e multi-view fusion for 3d object detection in lidar point clouds (2019 Waymo)
    * focus on how fusing different views of the same sensor can provide a model with richer information than a single view by itself

    * dynamic voxelization
        * same grouping stage
        * instead of sampling the points into a fixed number of fixed-capacity voxels, it preserves the complete mapping between points and voxels.
    
    * feature fusion
        * BEV, in which objects preserve their canonical 3D shape information and are naturally separable. it has the downside that the point cloud becomes highly sparse at longer ranges
        * perspective view, can represent the LiDAR range image densely, The shortcoming is that object shapes are not distance-invariant and objects can overlap heavily with each other in a cluttered scene.



* PIXOR: real time 3D object detection from point clouds (Mar 2019, Uber)


* STD: sparse 2 dense 3d object detection for point cloud(Jul 2019)

    * proposal generation module 
        * seed anchors for each point (16K)
        * Each anchor is associated with a reference box for proposal generation, with pre-defined size.
        * use a 3D semantic segmentation network to predict the class of each point and produce semantic feature for each point, then NMS to remove redundant anchors(500)
        * proposal generation network
            * gather 3D points within anchors for regression and classification
            * For points in an anchor, we pass their (X; Y;Z) locations, which are normalized by the anchor center coordinates, and semantic features from the segmentation network to a PointNet with several convolutional layers to predict classification scores, regression offsets and orientations.
            * Then compute offsets regarding anchor center coordinates (Ax;Ay;Az) and their pre-defined sizes (Al;Aw;Ah) so as to obtain precise proposals
    * PointsPool is applied for generating proposal features by transforming their interior point features from sparse expression to compact representation
        1. we randomly choose N interior points for each proposal with their canonical coordinates and semantic features as initial feature.
        2. using the voxelization layer to subdivide each proposal into equally spaced voxels
        3. apply Voxel Feature Encoding (VFE) layer with channels (128; 128; 256) [37] for extracting features of each voxel
        4. After getting 3D features of each proposal, we flatten them for following FC layers in the box prediction head.
    * box prediction
        * box estimation branch
            * use 2 FC layers with channels (512; 512) to extract features of each proposal.
            * Then another 2 FC layers are applied for classification and regression respectively.
        * IoU estimation branch 
            * an IoU estimation branch for predicting 3D IoU between boxes and corresponding ground-truth.
            * then multiply each box’s classification score with its 3D IoU as a new sorting criterion.
            * directly taking predicted 3D IoU as the NMS sorting criterion, performs not well.

    * summary: good work ! 
    * [zhihu]()




* 3d-ssd: point based 3d single stage object detector (Feb 2020)
    * point based methods:
        * Specifically, they are composed of two stages. In the first stage, they first utilize set abstraction (SA) layers for downsampling and extracting context features. Afterwards, feature propagation (FP) layers are applied for upsampling and broadcasting features to points which are discarded during downsampling. A 3D region  proposal network (RPN) is then applied for generating proposals centered at each point. Based on these proposals, a refinement module is developed as the second stage togive final predictions.
    * we first propose a novel sampling strategy based on feature distance, called F-FPS, which effectively preserves interior points of various instances.
    * we design a delicate box prediction network, which utilizes a candidate generation layer (CG), an anchorfree regression head and a 3D center-ness assignment strategy.
    * fusion sampling
    * box prediction network 
        * candidate generation layer 
            * initial center points shifed to candidate points, treated as new center points in CG layer
            * find surrounding points of each candidate point from the whole representative point set, concatenate their normalized locations and semantic features as input and apply MLP layers to extract features 
        * anchor-free regression head
        * 3d centerness assignment strategy 



* lidar rcnn(2021 Mar Tusimple)
    * we find an intriguing size ambiguity problem: Different from voxel-based methods that equally divide the space into fixed size, PointNet only aggregates the features from points while ignoring the spacing in 3D space

    * Assume that we already have a 3D LiDAR object detector that could generate lots of 3D bounding box proposals, our goal is to refine the 7DoF bounding box parameters (including size, position and heading) and scores of the proposals simultaneously

    * Size Ambiguity Problem
        * we should equip our LiDAR-RCNN with the ability to perceive the spacing and the size of proposals
    * [zhihu](https://zhuanlan.zhihu.com/p/359800738)



* CenterPoint(utexas 2021)
    * main underlying challenge in linking up the 2D and 3D domains lies in this representation of objects.
    * 2D CenterNet rephrases object detection as keypoint estimation. 
        * It takes an input image and predicts a w x h heatmap ^ Y for each of K classes 
        * To retrieve a 2D box, CenterNet regresses to a size map `S` shared between all categories. For each detection object, the size-map stores its width and height at the center location
    * CenterPoint 
        * stage1: predicts a class-specific heatmap, object size, a sub-voxel location refinement, rotation, and velocity.
            * heatmap head 
                * to produce a heatmap peak at the center location of any detected object.
                * increase the positive supervision for the target heatmap Y by enlarging the Gaussian peak rendered at each ground truth object center. （TODO)
            * regression heads 
                * a sub-voxel location refinement, reduces the quantization error from voxelization and striding of the backbone network.
                * height-above-ground, helps localize the object in 3D
                * yaw rotation angle, uses the sine and cosine of the yaw angle as a continuous regression target
                * box size
            * velocity head & tracking
                * learn to predict a two-dimensional velocity estimation for each detected object as an additional regression output, for tracking objects over time 
        * CenterPoint combines all heatmap and regression losses in one common objective and jointly optimizes them
        * stage2
            * only consider the four outward-facing box-faces together with the predicted object center. For each point, we extract a feature using bilinear interpolation from the backbone map-view outputM. Next, we concatenate the extracted point-features and pass them through an MLP 
            * predicts a class-agnostic confidence score and box refinement on top of one-stage CenterPoint’s prediction results

    * [centerPoint 源码分析](https://zhuanlan.zhihu.com/p/444447881)
    * [centerpoint 解读](https://zhuanlan.zhihu.com/p/524608535)

    * 总结： need code review !!! 


* 3d-MAN: 3d multi frame attention network for object detection (2021 google)

    * multi-frame fusion background: 
        * A straight-forward approach to fusing multi-frame point clouds is to use point concatenation
        * feature map level fusion: Fast & Furious
        * RNN based 
    * 3d single frame detection
    * 2d mutli-frame detection 

    * 3d-man:
        * fast single frame detector(FSD)
            * PointPillar with dynamic voxelization
        * memory bank 
            * proposal feature generation
            
        * multi-view alignment and aggregation module(MVAA)
            * To achieve this goal, the alignment stage has to figure out how to relate the identities of the proposals in the new frame to those in the stored frames
            * The cross-attention network is applied between the target frame and each stored frame independently with shared parameters, generating a feature vector for each target proposal (Vs) from each stored frame
            * cross view loss
                * The multi-view aggregation layer (MVAA-Aggregation) is responsible for combining these features from different perspectives together to form a single feature for each proposal
                * we use the new frame’s proposal features as the attention query inputs, and its corresponding extracted features in previous frames as the keys and values.
            * box prediction head

    * 总结： not well understood



* BEVDetNet: BEV LIDAR point cloud based 3d detection (2021 Jul)
    * we make use of BEV images as an efficient way of representing LiDAR data in a 2D grid.


* PV-RCNN: point-voxel feature set abstraction for 3d object detection(Apr 2021)

    * Generally, the grid-based methods are more computationally efficient but the inevitable information loss degrades the finegrained localization accuracy, while the point-based methods have higher computation cost but could easily achieve larger receptive field by the point set abstraction
    * the principle of PV-RCNN: the voxel-based operation efficiently encodes multi-scale featurerepresentations and can generate high-quality 3D proposals, while the PointNet-based set abstraction operation preserves accurate location information
    * The main challenge would be how to effectively combine the two types of feature learning schemes
    * pv-rcnn arch:
        * a 3D voxel CNN with sparse convolution as the backbone for efficient feature encoding and proposal generation.
            * 3d voxel cnn
                *  The network utilizes a series of 3  3  3 3D sparse convolution to gradually convert the point clouds into feature volumes
            * 3d proposal generation
                * by converting hte 3d feature volumes into 2d bev, high-quality 3D proposals are generated following the anchor-based approaches(second, pointPillars)
        * voxel-to-keypoint scene encoding
            * keypoints sampling
                * adopt the Furthest-Point-Sampling (FPS) algorithm to sample a small number of n keypoints
            * voxel set abstraction(VSA) 
                * to encode the multi-scale semantic features from the 3D CNN feature volumes to the keypoints.
            * extended VSA
                * extend the VSA module by further enriching the keypoint features from the raw point clouds P and the 8 downsampled 2D bird-view feature maps
            * predicted keypoint weighting
                * by checking whether each key point is inside or outside of a ground-truth 3D box  
        * point-to-grid RoI feature abstraction        
            * roi grid pooling via set abstraction 
                * to aggregate the keypoint features to the RoI-grid points with multiple receptive fields.
                * After obtaining each grid’s aggregated features from its surrounding keypoints, all RoI-grid features of the same RoI can be vectorized and transformed by a two-layer MLP with 256 feature dimensions to represent the overall proposal.
        * 3d proposal refinement and confidence prediction


* PV-RCNN++ (2022 Nov)
    * background: 
        * 3d object detection with 2d images 
            * image-based 3D detection methods suffer from inaccurate depth estimation and can only generate coarse 3D bounding boxes
            * scene understanding with surrounding cameras: depth-based implicit projection to project image features to BEV space
        * representation learning on point clouds
        * 3d object detections with point clouds
            * exploring dierent feature grouping strategy for generating 3D boxes, .e.g 3dssd
    
    * sectorized proposal-centric sampling for efficient and representative keypoint sampling 
        * as keypoints bridge the point-voxel representations
        * proposal-centric filtering
        * sectorized keypoint sampling
            * we divide proposal-centric point set P0 into s sectors centered at the scene center
            * we divide the task of sampling n keypoints into s subtasks of sampling local keypoints
    * local vector representation for structure-preserved local feature learning
        * vectorPool aggregration on point clouds
            * generate position-sensitive local features by encoding different spatial regions with separate kernel weights and separate feature channels, which are then concatenated as a single vector representation to explicitly represent the spatial structures of local point features.
        












## 点云语义分割

* [zhihu: 激光雷达，点云语义分割](https://zhuanlan.zhihu.com/p/412161451)

    * 语义分割
    
    * 实例分割
        * 自顶向下： LidarSeg，物体检测是第一步
        * 自下向上： 实例分割是通过对底层语义分割结果进行聚类得到的



## 点云跟踪

* [3D点云数据 目标检测和跟踪任务小结](https://zhuanlan.zhihu.com/p/436519462)




## 3D视觉感知 

* [视觉感知：3D感知算法](https://zhuanlan.zhihu.com/p/426569335)
    * 图像反变换
    * 提取图像关键点映射3D模型
    * 2D/3D几何约束
    * 直接生成3D物体框
        * 前三类都是从2D图像出发，有的将图像变换到BEV视图，有的检测2D关键点并与3D模型匹配，还有的采用2D和3D物体框的几何约束
        * 还有一类方法从稠密的3D物体候选出发，通过2D图像上的特征对所有的候选框进行评分，评分高的候选框既是最终的输出
        * SS3D
            * 这里的3D物体框并不是一般的9D或7D表示（这种表示很难直接从图像预测），而是采用更容易从图像预测也包含更多冗余的2D表示，包括距离（1-d），朝向（2-d，sin和cos），大小（3-d），8个角点的图像坐标（16-d）。再加上2D物体框的4-d表示，一共是26D的特征。所有这些特征都被用来进行3D物体框的预测，预测的过程其实就是找到一个与26D特征最为匹配3D物体框
        * FCOS3D
    * 深度估计
        * 单阶段3D物体检测网络中大多都包含了深度估计的分支
        * 自动驾驶感知还有另外一个重要任务，那就是语义分割。语义分割从2D扩展到3D，一种最直接的方式就是采用稠密的深度图，这样每个像素点的语义和深度信息就都有了 ？？？ why need 语义分割？
        * 深度估计与语义分割任务有着相似之处 ？？
        * 双目深度估计 

* [最新论文：BEV感知综述、评估和对策](https://zhuanlan.zhihu.com/p/565212506)

* [多传感器融合综述: fov vs bev](https://www.guyuehome.com/38657)
* [笔记：vision centric bev perception: survey](http://www.guyuehome.com/41737)
* [自动驾驶之心: bev感知算法综述](https://mp.weixin.qq.com/s/en8k_EpIaWy0s6wRg2MqYg)



* survey: 


* DETR3D: 3d object detection from multi-view images via 3d-to-2d queries (2021)
    * extracts 2D features from multiple camera images and then uses a sparse set of 3D object queries to index into these 2D features, linking 3D positions to multi-view images using camera transformation matrices (top-down)
    * back-project a set of reference points decoded from these object priors to each camera and fetch the corresponding image features extracted by a ResNet backbone [8]. The features collected from the image features of the reference points then interact with each other through a multi-head self-attention layer
    * arch
        * feature learning 
        * top-down detection head
            1. predict a set of bbox centers associated with object queries 
            2. project these centers into all feature maps with camera transformation matrices 
            3. sample feature via bilinear interpolation and incorporate them into object queries
            4. object interactions using MHA 

* BEVFormer: learning BEV representation from multi-camera images via spatio-temporal transormers (2022)
    * arch
        * feed multi-camera images to 2d backbone -> extract features of different camera view 
        * BEV queries are grid-shaped learnable parameters
        * preserv BEV features at priori timestamp `t-1` 
        * at each encoder layer
            * first, use BEV queries to query temporal info via temporal self-attention
            * then, use BEV queries to inquire spatial info via multi-camera via spatial cross-attention 
        


## 3D 点云 image融合 + 时序features 

* [zhihu: 无人驾驶中视觉与雷达多传感器如何融合?](https://www.zhihu.com/question/453174887)
    * proposal融合：e.g.  mv3d, avod 
        * 以物体候选框为中心来融合不同的特征，融合的过程中一般会用到ROI pooling（比如双线性插值），而这个操作会导致空间细节特征的丢失。 
    * 特征层融合, e.g. pointPaiting, 



* Frustum PointNets for 3d detection from rgb-d(Apr 2018)
    * one key challenge: how to efficiently propose possible locations of 3D objects in a 3D space
    * frustum proposal 
        * With a known camera projection matrix, a 2D bounding box can be lifted to a frustum (with near and far planes specified by depth sensor range) that defines a 3D search space for the object.
        * collect all points within the frustum to form a frustum point cloud
        * normalize the frustums by rotating them toward a center view such that the center axis of the frustum is orthogonal to the image plane
    * 3D Instance segmentation
        * we realize 3D instance segmentation using a PointNet-based network on point clouds in frustums.
    * amodal 3d box estimation
        * estimates the object’s amodal oriented 3D bounding box by using a box regression PointNet together with a preprocessing transformer network
    * [zhihu](https://zhuanlan.zhihu.com/p/41634956)


* MV3D: mutli-view 3d object detection network for av (2017 Jun)
    * The main idea for utilizing multimodal information is to perform region-based feature fusion
    * 3d point cloud representation
        * BEV & Front view 
    * 3d proposal network
        * Given a bird’s eye view map. the network generates 3D box proposals from a set of 3D prior boxes
    * region based fusion network 
        * combine features from multiple views and jointly classify object proposals and do oriented 3D box regression
        * mutli-view ROI pooling:
            * ROI pooling [10] for each view to obtain feature vectors of the same length.
            * Given a 3D proposal p3D, we obtain ROIs on each view
        * deep fusion
    * oriented 3d bbox regression


* AVOD: joint 3d proposal generation and object detection from view aggregation(Jul 2018)
    * most state-of-the-art deep models for 3D object detection rely on a 3D region proposal generation step for 3D search space reduction
    * background:
        * hand crafted features for 3d proposal generation 
        * proposal free, single shot detectors 
        * monocular-based proposal generation, e.g.  F-PointNet
            * Any missed 2D detections will lead to missed 3D detection
        * monocular-based 3d detectors 
        * 3d region proposal networks
            * MV3D extends the image based RPN to 3D by corresponding every pixel in the BEV feature map to multiple prior 3D anchors.
            * These anchors are then fed to the RPN to generate 3D proposals that are used to create view-specific feature crops from the BEV, front view of [3], and image view feature maps.
    * avod arch:
        * generating feature maps from point clouds and images 
            * generate a six-channel BEV map from a voxel grid representation of the point cloud at 0:1 meter resolution.
        * feature extractor 
        * multimodal fusion region proposal network
            * To generate the 3D anchor grid, (tx; ty) pairs are sampled at an interval of 0:5 meters in BEV, while tz is determined based on the sensor’s height above the ground plane.
        * extracting feature crops via multiview crop and resize operations
            * Given an anchor in 3D, two regions of interest are obtained by projecting the anchor onto the BEV and image feature maps. The corresponding regions are then used to extract feature map crops from each view
        * dimensionality reduction via 1x1 conv 
        * 3d proposal generation    
            * using fused the equal-sized feature maps from BEV and image feature maps to regress axis aligned object proposal boxes and output an object/background “objectness” score
        * 3d bbox encoding
            * explicit orientation vector regression
        * [zhihu](https://zhuanlan.zhihu.com/p/40271319)


* PointFusion: Deep sensor fusion for 3d bbox estimation(2018 Apr)
    * performs 3D bounding box regression from a 2D image crop and a corresponding 3D point cloud that is typically produced by lidar sensors
    * fusion network:
        * input: image feature + corresponding point cloud features 
        * output: 3d bbox 
        * global fusion network:
            * a concatenation of the two vectors, followed by applying a number of fully connected layers, results in optimal performance
            * drawback: the variance of the regression target is directly dependent on the particular scenario
        * dense fusion network
            * Instead of directly regressing the absolute locations of the 3D box corners, for each input 3D point we predict the spatial offsets from that point to the corner locations of a nearby box.
            * For each point, these are concatenated with the global PointNet feature and the image feature resulting in an n  3136 input tensor.
        * dense fusion prediction scoring
    * [zhihu](https://zhuanlan.zhihu.com/p/42111500)
        * AVOD将RGB和BEV图像经过特征提取后进行fusion，结合了颜色信息与空间分布信息，但是使用的BEV是经过点云投影得到，存在空间信息的损失；F-PointNet使用raw point cloud提取空间几何特征，没有任何信息的损失，但是没有充分利用RGB信息。而PointFusion权衡了二者的利弊，使用raw point cloud的同时辅以颜色信息。


* MVX-net: multimodal voxelnet for 3d object detection (2019)
    * augment LiDAR points with semantic image features and learn to fuse image and LiDAR features at early stages for accurate 3D object detection
    * multimodal fusion
        * pointFusion
            * an early fusion technique where every 3D point is aggregated by an image feature
        * VoxelFusion
            * relatively later fusion strategy where the features from the RGB image are appended at the voxel level
            * every non-empty voxel is projected onto the image plane to produce a 2D region of interest (ROI).
            * the image features within the ROI are pooled to produce a 512-dimensional feature vector
    * shit !!





* ContFuse: continous fusion for multi-sensor 3d object detection(2020 Dec, Uber)
    * continuous fusion layer
        * Given the input camera image feature map and a set of LIDAR points, the target of the continuous fusion layer is to create a dense BEV feature map where each discrete pixel contains features generated from the camera image, then fused with BEV feature maps extracted from LIDAR.
        * one difficulty of image-BEV fusion is that not all the discrete pixels on BEV space are observable in the camera.
            1. for each target pixel in the dense map, find its nearest K LIDAR points over the 2D BEV plane using Euclidean distance.
            2. exploit MLP to fuse information from these K nearest points to \interpolate" the unobserved feature at the target pixel.
            3. ?
    * [ZHIHU](https://zhuanlan.zhihu.com/p/45338728)



* pointPainting(2020 nuTonomy)
    * while a lidar point cloud can trivially be converted to bird’seye view, it is much more difficult to do so with an image  --> hence,  a core challenge of sensor fusion network design lies in consolidating the lidar bird’s-eye view with the camera view. 

        * object centric fusion: e.g. MV3D, AVOD
            * fusion happens at the object proposal level by applying roi-pooling in each modality from a shared set of 3D proposals

        * continuous feature fusion, e.g. ContFuse
            * shared features across all strides of images and lidar backbones, but requiring a mapping(a priori) for each sample from point cloud to image 

        * explicit transform 
            * explicit transform image to BEV, then fusing in BEV
        
        * detection seeding
            * semantics are extracted from an image a priori and used to seed detection in the point cloud

    * PointPainting architecture accepts point clouds and images as input and estimates oriented 3D boxes

        * stage1: image based semantic segmentation
        * stage2: fusion
            * transformation ops:
                * Kitti transformation； T(camera->lidar)
                * nuScenes transformation:  T = T(ego->camera) * T(ego_l -> ego_c) * T(lidar -> ego)
                    *  lidar frame to the ego-vehicle frame; ego frame at time of lidar capture, `tl` to ego frame at the image capture time, `tc`; and ego frame to camera frame 
        * stage3: lidar 3d detection 
    
    * 总结：通过图像语义信息增加原始点云


* TransFusion: robust lidar-camera fusion for 3d object detection with transformers(2022 cvpr)
    * background:
        * output level fusion: F-PointNet
        * proposal level fusion, mv3d, avod
        * These coarse-grained fusion methods show unsatisfactory results since rectangular regions of interest (RoI) usually contain lots of background noise
        * point-level fusion
            1. find hard association between LiDAR points and image pixels based on calibration matrices
            2. then augment LiDAR features with the segmentation scores [47, 52] or CNN features[10, 22, 40, 48, 63] of the associated pixels through point-wise concatenation.
            3. [or] project a point cloud onto the bird’s eye view (BEV) plane and then fuse the image features with the BEV pixels
        * performance degrades seriously with low-quality image features and hard association issues
    * Our key idea is to reposition the focus of the fusion process, from hard-association to soft-association, leading to the robustness against degenerated image quality and sensor misalignment.
    * arch:
        * query initialization
            * we propose an input-dependent initialization strategy based on a center heatmap to achieve competitive performance using only one decoder layer
            * The  positions and features of the selected candidates are used to initialize the query positions and query features.
        * category-aware
            * we element-wisely sum the query feature with a category embedding, produced by linearly projectingthe one-hot category vector into a Rd vector
        * transformer decoder & FFN 

        * lidar camera fusion
            * image feature fetching 
                * When an object only contains a small number of LiDAR points, it can fetch only the same number of image features, wasting the rich semantic information of high-resolution images
                * solution: retain all the image features FC ∈ RNv×H×W×d as our memory bank, and use the cross-attention mechanism in the transformer decoder to perform feature fusion in a sparseto-dense and adaptive manner
            * SMCA for image feature fusion
                * first identify the specific image in which the object queries are located using previous predictions as well as the calibration matrices
                * then perform cross attention between the object queries and the corresponding image feature map.
                * design a spatially modulated cross attention (SMCA) module, which weighs the cross attention by a 2D circular Gaussian mask around the projected 2D center of each query.
                * In this way, each object query only attends to the related region around the projected 2D box,
            


* BEVFusion: multi-task mutli-sensor fusion with unified bev representation (Jun 2022, MIT)
    * background:
        * cameras capture data in perspective view and LiDAR in 3D view. To resolve this view discrepancy, we have to find a unified representation that is suitable for multi-task multi-modal feature fusion
        * LiDAR-to-camera projection, introduces severe geometric distortion, less effective for geometric-oriented tasks, such as 3D object recognition.
        * camera-to-LiDAR projection, by augment the LiDAR point cloud with semantic labels, CNN features or virtual points from 2D images, barely work on semantic-oriented tasks, such as BEV map segmentation
        * multi-sensor fusion
            * proposal-level: creates object proposals in 3D and projects the proposals to images to extract RoI features
            * point-level: paint image semantic features onto foreground LiDAR points and perform LiDAR-based detection on the decorated point cloud inputs
    * transform multi-modal features into a unified BEV representation that preserves both geometric and semantic information. and accelerate BEV pooling with precomputation and interval reduction, then apply the convolution-based BEV encoder to the unified BEV features
    * unified representation: BEV
    * efficient camera to BEV transformation 
        1. Following LSS [39] and BEVDet [20, 19], we explicitly predict the discrete depth distribution of each pixel
        2. then scatter each feature pixel into D discrete points along the camera ray and rescale the associated features by their corresponding depth probabilities -> a camera feature point cloud of size NHWD
        3. quantized camera feature cloud along the x; y axes with a step size of r(0.4m) 
        4. BEV pooling operation to aggregate all features within each r x r BEV grid and flatten the features along the z-axis
    * BEV pooling precomputation
        1. associate each point in the camera feature point cloud with a BEV grid
        2. as the coordinates of the camera feature point cloud are fixed, precompute the 3D coordinate and the BEV grid index of each point
        3. sort all points according to grid indices and record the rank of each point.
    * Interval reduction
        * aggregate the features within each BEV grid by some symmetric function
        * a reduction kernel: we assign a GPU thread to each grid that calculates its interval sum and writes the result back (500ms -> 2ms)
    

* LIFT: learning 4D lidar image fusion transformer for 3d object detection (alibaba 2022)
    * LIFT learns to align the input 4D sequential cross-sensor data to achieve multi-frame multi-modal information aggregation.
    * temporal fusion
        * way1: directly concatenate points from adjacent frames [2, 20, 35], but without explicit consideration of temporal correlation 
            * 4d-net 
            * Second 
        * way2: [8,24,41,42] fusion at feature level
            * 3d-man
            * 42: lidar based online 3d video object detection with graph-based message
            * Offboard 3d object detection from point cloud sequences
            * 8: an LSTM approch to temporal 3d object detection 
            * object-centric desin  [24, 36, 41] 
                * temporal feature fusion is conducted on top of object proposals 
                * 36: auto4d 
            * scene-centric design 
                * fusion based on whole scene 
            * RNN based
                * computationaly intensive 
    * cross-sensor fusion
        * fusion based on 2d results 22
            * 22: Frustum pointnets for 3d object detection from rgb-d data (2018)
        * fusion based on object region proposals 3, 11
            * 11: Joint 3d proposal generation and object detection from view aggregation(2018)
        * fuse cross-modal features at BEV space, 14, 15, 44 
            * 14: multi-task, multi sensor fusion for 3d object detection (2019 cvpr)
            * 15: deep continuous fusion for multi-sensor 3d object detection (2018)
            * 44: 3d-cvf: generating joint camera and lidar features using cross-view spatial feature fusion for 3d object detection (2020)
        * fusion at point level,  9, 27, 31, 47 
            * 47: cross-modal 3d object detection and tracking for auto driving 
            * 31: pointPainting
            * 27: multimodal voxelnet for 3d object detection (2019)
            * 9: Epnet: enhancing point features with image semantics for 3d object detection (2020)
        * those projectionbased approaches are easily affected by projection errors, resulting in ambiguous fusion with misaligned information

    * lidar image fusion transformer(LIFT)
        * takes both sequential point clouds and images as input and aims at exploiting their mutual interactions.
        * grid feature encoder : process input sequential cross-sensor data into grid features 
            * camera feature fetching   
            * pillar feature extraction
            * point-wise attention
                
        * sensor time 4D attention: learn 4D sensor-time interaction relations given the grid-wise BEV representations
            * sparse window partition

        * shit !!!


* DeepFusion: Lidar-Camera deep fusion for mutli-modal 3d object detection (2022 google)
    * a key challenge in fusion is how to effectively align the transformed features from two modalities.
    * we fuse deep camera and lidar features instead of decorating raw lidar points at the input level so that the camera signals do not go through the modules designed for point cloud
    * To address the alignment issue caused by geometryrelated data augmentation, we propose `InverseAug`, 
    * LearnableAlign
        * the input contains a voxel cell, and all its corresponding N camera features


* DETR4D: direct multi-view 3d object detection with Sparse attention 
    * heatmap based query initialization
    * projective cross-attention
    * temporal modelin 





## 真值标注 

* auto4D learning to label 4d object from sequential point clouds (2021)

    * 



* MPPNet 











## 点云预处理 background 

* [zhihu: 点云预处理](https://zhuanlan.zhihu.com/p/455810371)

    * 点云滤波 

    * 点云关键点

    * 特征与特征描述

    * [点云配准](https://zhuanlan.zhihu.com/p/104735380)

    * 点云分割与分类 

* [github: pcl-learning](https://github.com/HuangCongQing/pcl-learning)

* [github: PCL-notes](https://github.com/MNewBie/PCL-Notes)

* [zhihu: 3d点云基础](https://zhuanlan.zhihu.com/p/344635951)

    * 点云内容：包括三维坐标（XYZ）和激光反射强度（Intensity），强度信息与目标的表面材质、粗糙度、入射角方向以及仪器的发射能量、激光波长有关
    * 点云属性：空间分辨率、点位精度、表面法向量等。
    * 点云存储格式：
        * pts, 直接按 XYZ 顺序存储点云数据
        * las, 激光雷达数据. C, F, T, I, R, N, A, RGB
        * PCD，具有文件头，用于描绘点云的整体信息：定义数字的可读头、尺寸、点云的维数和数据类型；一种数据段，可以是 ASCII 码或二进制码
        * .xyz，前面 3 个数字表示点坐标，后面 3 个数字是点的法向量，数字间以空格分隔
        * .pcap 是一种通用的数据流格式，现在流行的 Velodyne 公司出品的激光雷达默认采集数据文件格式 
    * 点云表示：
        * 将无序的空间点转变为规则的数据排列
            * 投影为2D图像
            * 转换为3D voxel
        * 直接使用原始点云
    * 基于点云的分类
        * 通常是先通过聚合编码器生成全局嵌入，然后将嵌入通过几个完全连通的层来获得最终结果。
        * 基于投影的方法：
            * multi-view representation
            * volumetric representation
        * 基于点的方法：
    

* [pcl org](https://pointclouds.org/)
* [official pcl cuda support branch](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.11.0/doc/tutorials/content/gpu_install.rst)  :: too old 
* [cuPCL-x86 branch](https://github.com/NVIDIA-AI-IOT/cuPCL/tree/x86_64_lib/cuICP)


* [PCL 学习指南](https://zhuanlan.zhihu.com/p/268524083)
    * 实现点云相关的获取、滤波、分割、配准、检索、特征提取、识别、追踪、曲面重建、可视化等

* [点云配准](https://zhuanlan.zhihu.com/p/104735380)

    * 迭代最近点算法（Iterative Closest Point, ICP
    * ICP 一般算法流程为：
        1. 点云预处理 - 滤波、清理数据等
        2. 匹配 - 应用上一步求解出的变换，找最近点
        3. 加权
        - 调整一些对应点对的权重
        4. 剔除不合理的对应点对
        5. 计算 loss
        6. 最小化 loss，求解当前最优变换
        7. 回到步骤 2. 进行迭代，直到收敛






### cuPCL/ICP code review 


```c++
// Iter_para [PCountN=35847, Maxiterate, threshold, acceptrate, distance_threshold, relative_mse]
Iter_para  iter{50, 1e-6, 1.0, 0.5, 0.0001} 

```


* [pclVisualizer](https://pcl.readthedocs.io/projects/tutorials/en/latest/pcl_visualizer.html)

    * depends [vtk](https://vtk.readthedocs.io/en/latest/index.html)

* [open3d pcl visualization](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)




### 点云稠密化

* [multimodal virtual point 3d detection](https://tianweiy.github.io/mvp/)
    * Our approach takes a set of 2D detections to generate dense 3D virtual points to augment an otherwise sparse 3D point-cloud

* [sparse to dense 3d object detector for point cloud](https://arxiv.org/abs/1907.10471)
    * 使用原始点云作为输入，通过为每个点播种新的球形anchor来生成准确的proposal

* [sparse fuse dense: 3d detection with depth completion](https://arxiv.org/abs/2203.09780)


## 点云标注

* [点云标注工具](https://zhuanlan.zhihu.com/p/402530977)

* 3D点云连续帧标注

* [aws SageMaker: Data Labeling](https://aws.amazon.com/sagemaker/)
    * [doc: sageMaker for lidar point](https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/sms-point-cloud.html)

* [segments.ai](https://segments.ai/point-cloud-labeling)
* [scale.ai](https://scale.com/3d-sensor-fusion)
* [dataloop.ai](https://dataloop.ai/platform/lidar/)


* [point cloud labelling tool](https://github.com/supervisely-ecosystem/pointcloud-labeling-tool)

* [自动驾驶标注工具](https://zhuanlan.zhihu.com/p/56514817)






## lidar based open project 

* [cvml](https://github.com/darylclimb/cvml_project)

* [openPCDet](https://github.com/open-mmlab/OpenPCDet)

* [xtreme1](https://github.com/xtreme1-io/xtreme1) :  1st open-source platform for multi-sensory training data 

* [toyota research: github: tri-ml/vidar](https://github.com/TRI-ML/vidar)




#### 源码

* SECOND 

* CenterPoint 

* PointPillar

* [github: relation networks for object detection](https://github.com/msracver/Relation-Networks-for-Object-Detection)




## issues 

* 多相机 4d vector space标注的

* 自动标注（4D Auto-lableing)

*  如何把新的感知框架工程化部署到车端

* 离线4D真值系统的自动化

* 半监督/仿真/ 4d场景重建（最近很火的NeRF） 在 多相机BEV架构下的应用，也需要不断关注


* [数据闭环与autolabelling 方案总结](https://zhuanlan.zhihu.com/p/587140851)


* [数据闭环的核心： auto-labeling 方案分享](https://zhuanlan.zhihu.com/p/533907821)

    * Robesense RS-reference 系统

    * 3d 动态元素自动标注

    * TODO: 作为算法工程师的我们在角落瑟瑟发抖， view transformer 如何部署？ BEV 真值从何而来？ 传感器时空标定精度是否足够？云端服务器8卡A100完全不够大家开发？

    * 如何离线低成本的获得bev 任务需要的真值(3d 目标，速度加速度， 3d车道线， 路沿， 可行驶区域等)
    * 如何工程化的量化评估车端bev 的感知性能

* [zhihu: Tesla AI DAY 深度分析 硬核:  auto-labeling](https://zhuanlan.zhihu.com/p/466426243)

    * 4D Space + Time labelling -> 其实像一个vector space 下 3D 标注 + 时间序列，加入时间序列主要作用是知道前面发生了什么，把前面的东西保留，可以将信息投到后面  --> 在3D空间下标注，然后再投到8个摄像机里面


* [自动驾驶中的自动标注](https://zhuanlan.zhihu.com/p/113749235)





## 工程问题

* bev 真值系统
    
    * 

* corner case 挖掘系统 





## 目标检测算法

* [CenterNet](https://www.bilibili.com/video/BV1r44y1a75j/?spm_id_from=333.337.search-card.all.click&vd_source=b2ac0b748d460a55f64f7da95565f7ef)


* [cvpr 2021 论文拍点 - 3d 目标检测](https://zhuanlan.zhihu.com/p/389319123)

