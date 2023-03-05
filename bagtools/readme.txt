0.  查看bag情况,这里要注意的是, gazebo采集数据不能加速,否则会掉帧(rosbag record情况). kitti 格式需要预处理nan值.
    需要通过rosbag info xxx.bag 看看具体有多少帧数据,对比生成的结果.以及确认包的好坏.
    pc = pc[~np.isnan(pc).any(axis=1)]  # remove column with nan
1.  roscore
2.  (new terminal) rosbag play floor2_2023-02-27-16-48-26.bag --pause --start 0 --rate 1 
3.  
    修改 sim_save.py的 line21 root到指定存储目录
    root = "/media/kid/info/map/savedframes/garage"
4.  (new terminal) 使用系统自带的默认 python2 (崔家赫修改过)
    python sim_save.py # 进行拆包
5.  python gt_mapping.py 选取若干帧点云叠加查看,拆包后的地图结果.



语义类别(仿真没有建天花板):
    garage: 这个场景相对比较空,没有动态目标和其他复杂的东西. floor, wall, pillar 三个类别.
    floor2: 左顺把场景内的物品都剔除了. 所与差不多也是. 没有柱子,有一个桌子.
    office: 虽然是矩形,但是内部有不少零碎的物品.需要着重定义下.
   

修改point labeler的标签定义
/home/kid/catkin_ws/src/point_labeler
color map参考:
    0: [0, 0, 0],  # "unlabeled", and others ignored
    1: [0, 0, 255],  # outliner
    10: [245, 150, 100],  # "floor"
    11: [245, 230, 100],  # "wall"
    12: [150, 60, 30],  # "pillar"
    13: [30, 30, 255],  # "celling"
    21: [180, 30, 80],  # "desk"
    22: [255, 0, 0],  # "chair"



