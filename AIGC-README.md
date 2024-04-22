

## UniSim


1. [GAIA-1 自主生成AI 模型]()



## 合成数据


1. [AI自给自足！用合成数据做训练，效果比真实数据还好丨ICLR 2023](https://zhuanlan.zhihu.com/p/608269775)

2. [微软研究员联合Yoshua Bengio推出AIGC数据生成学习范式Regeneration Learning](https://zhuanlan.zhihu.com/p/620849058)

    * 数据生成任务需要刻画数据的整体分布而不是抽象表征，需要一个新的学习范式来指导处理数据生成的建模问题
    * 对于数据理解任务，X 通常比较高维、复杂并且比 Y 含有更多的信息，所以任务的核心是从 X 学习抽象表征来预测 Y。因此，深度学习中非常火热的表征学习（Representation Learning，比如基于自监督学习的大规模预训练）适合处理这类任务。
    * 对于数据生成任务，Y 通常比较高维、复杂并且比 X 含有更多的信息，所以任务的核心是刻画 Y 的分布以及从 X 生成 Y。
    * 深度生成模型（比如对抗生成网络 GAN、变分自编码器 VAE、自回归模型 AR、标准化流模型 Flow、扩散模型 Diffusion 等）
    * 相比于直接从 X 生成 Y，Regeneration Learning 先从 X 生成一个目标数据的抽象表征 Y’，然后再从 Y’ 生成 Y。

3. [合成数据的大规模应用要来了吗](https://zhuanlan.zhihu.com/p/594209122)

    * 新的工具
        1. UE5的发布， 光追等技术让更真实的图片成为可能， 当然也包括人类的创建更加容易了。
        2. Unity 的Perception 1.0, 专门为合成数据的生产做了适配
        3. Nvidia 的 Ominiverse 套件， 也包括了很多合成数据的内容
    * 新的研究
        1. 生成式模型的进展， Diffusion Model 在生成式大模型的扩展性让之前GAN 的任务再次摆上台面， 例如 InstanceMap 2 Photorealistic ， Video Generation， Domain Transfer， Sim2Real
        2. 从单一图片或者多视图中得到的3D 表达， 这里关于NeRF， 关于可微分渲染器 的发展都有了很大的进步
        3. 身体，姿势， 脸部的表达与生成
        4. 场景的隐式理解

    * [Synthesis AI: New use case for synthetic data](https://synthesis.ai/2022/07/14/cvpr-22-part-ii-new-use-cases-for-synthetic-data/)
    
    * [Synthetic AI: Digital humans](https://synthesis.ai/2022/08/08/cvpr-22-part-iii-digital-humans/)

    * [Synthetic AI: SDG from CVPR22](https://synthesis.ai/2022/09/07/cvpr-22-part-iv-synthetic-data-generation/)


* [CVPR 2023: Synthetic Data for ADAS](https://www.youtube.com/playlist?list=PLbQ8cd-A-UTBu4ndxinpUQNyH1BMxpqzR)

* [如何解决自动驾驶感知的长尾问题？ 合成数据会是答案吗](https://zhuanlan.zhihu.com/p/542686085)

* [自动驾驶长尾难题解法 --- Nvidia DriveSim NRE](https://zhuanlan.zhihu.com/p/568044393)


* [自动驾驶长尾问题解法 --- GET3D](https://zhuanlan.zhihu.com/p/574143176)

* [Nvidia StyleGAN]()

* [Anyverse.AI hyperspectral synthetic data platform for adas](https://anyverse.ai/)

* [nvdiffrec](https://nvlabs.github.io/nvdiffrec/)

* [nv Kaolin](https://github.com/NVIDIAGameWorks/kaolin)

* [nv instant-ngp](https://github.com/NVlabs/instant-ngp)

* [wabbi: GeoSim](https://tmux.top/publication/geosim/)

* [unity: GeomSimCities](https://geosimcities.com/)

* [GeoSim 作者解读: Camera simulation](https://zhuanlan.zhihu.com/p/377570852)

* [GeoSim 厘米解读]()

* [什么样的自动驾驶仿真测试系统能够帮助自动驾驶技术推进，仿真可以为自动驾驶技术落地提供哪些帮助？](https://www.zhihu.com/question/497338145/answer/2227266369)


## AIGC 

1. [万字长文： AIGC技术与应用解析](https://zhuanlan.zhihu.com/p/607822576)

    * 基础模型：
        1. VAE
        2. GAN
        3. Diffusion Model
            * DALL-E 2
            * Imageen
            * Stable Diffusion
            * 
        4. ViT 
    
    * 预训练大模型
        1. Florence(swint v2)
        2. NLP: PaLM, ChatGPT 
        3. multi-mode: CLIP,  GLIP, Stable Diffusion


2. [AIGC图像生成模型发展与高潜方向](https://zhuanlan.zhihu.com/p/612856195)


3. [Synthesis ai: synthetic data guide](https://synthesis.ai/synthetic-data-guide/)
        1. CGI approach
        2. Generative AI and diffusion models 
        3. NeRF
        4. Bridging the domain gap with GANs

4. [nv toronto ai: research topics](https://research.nvidia.com/labs/toronto-ai/)





## GAN

1. styleGAN

2. cycleGAN

3. [GAN vs Diffusion models](https://www.sabrepc.com/blog/Deep-Learning-and-AI/gans-vs-diffusion-models)

4. [6 GAN you should know](https://neptune.ai/blog/6-gan-architectures)





## stable diffusion

1. [illustrated stable diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)


## uniSim

1. [blog](https://waabi.ai/unisim/)
2. [climateNerf](https://climatenerf.github.io/)


## Nerf

1. [zhihu: nerf及其发展](https://zhuanlan.zhihu.com/p/512538748)

2. [nerf at cvpr2023](https://markboss.me/post/nerf_at_cvpr23/)

* paper with code
    1. [neural scene graph for dynamic scenes](https://github.com/princeton-computational-imaging/neural-scene-graphs)
    2. [free view synthesis](https://github.com/isl-org/FreeViewSynthesis)
    3. [category specific mesh reconstruction from image collections](https://github.com/akanazawa/cmr)
    4. [pointRend](https://github.com/zsef123/PointRend-PyTorch)


## end2end 

1. [PnPnet](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/pnpnet.html)
    * MOT has two challenges: the discrete problem of data association and the continuous problem of trajectory estimation.


## 3d auto labeling 

1. [mppnet](https://github.com/open-mmlab/OpenPCDet)

2. 