# lerobot-ros2
lerobot 切换到 ros2 

## 规划
- sim（集成 Issac sim）
- benchmark
- lerobot
    - model 训练
        - 使用已有 dataset
    - model 推理
        - 使用设备数据
        - 使用已有 dataset
    - 设备数据抓取
    - 爪子能够挪到小车上？
- 抽屉式迁移，尽量能做到和 lerobot 沟通，一步一步 ros 化
- 需要在硬件控制这一层添加一个

## 步骤
- [X] lerobot 功能总结
- [X] lerobot 依赖分析
- [ ] lerobot 模块与机理分析
- [ ] lerobot-ros2 规划（考虑泛用性）

# lerobot 
## 功能
1. 下载 dataset 以及预览 - hold
2. 下载训练好的模型 - hold
3. 训练模型
    - 使用已有的 dataset
4. 推理
    - 使用已有的 dataset
    - 使用设备数据 - focus

## 依赖
参考 python_dependencies.md



# troubleShooting:
1. 下载模型时报错 network connection timeout
   解决方法：设置代理镜像：export HF_ENDPOINT=https://hf-mirror.com
            解决不了还可参考：https://github.com/huggingface/transformers/issues/17611

2. 运行 data visualizer 报错：
    ValueError: No valid stream found in input file. Is -1 of the desired media type?
    解决方法：降级 ffmpeg：conda install -c conda-forge ffmpeg=6.1.1 -y
            可能也有其他解决方案，可参考：https://github.com/huggingface/lerobot/issues/970

3. 运行 gazebo 报错 ：
    FileNotFoundError: [Errno 2] No such file or directory: 'gz'
    解决方法：gazebo 未安装，运行 sudo apt install ros-"distro"-gazebo-ros-pkgs 
    其中 distro 换成自己的 ros 发行版（ubuntu 22.02 是 humble）

4. 注意 so-arm-101 package.xml 里面有被注释掉的行

5. 运行 gazebo 报错：
    ERROR： gz sim -r empty.sdf 运行失败
两种可能：
    - gazebo 是版本 6（fortress），那么需要将 "gz sim" 命令换成等价的 "ign gazebo"
    - gazebo 是版本 7（gardon）及 以上 需要 apt install 安装 libgz-sim<一个数字> 依赖。具体数字就是 gazebo 的版本。。安装 gazebo 教程 https://gazebosim.org/docs/latest/ros_installation/ 。安装 libgz 教程：https://gazebosim.org/api/sim/9/install.html
    p.s. 根据 gazebo 官方说法，humble 对应的应该是 gazebo 6（fortress），但是这个仓里面使用的是 gazebo 7 （gardon）。