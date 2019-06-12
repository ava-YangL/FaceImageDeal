
1 get_facepoints_facepp.py, 通过face++的API获得106个关键点，但这些关键店不包括额头
2 warp.py 通过另一个大神的关键店检测算法互动儿额头的关键点，与1中的关键店合并，可通过两张图片的关键点，对人脸进行Warp
3 getMask.py 获得与关键点吻合的Mask，也是对大神代码的改进，其中对Mask进行了腐蚀与膨胀
4 SmoothDepth文件，包含了对图像的均值滤波等平滑操作
5 webcam_record.py 是大神原来的代码
6 关键点检测模型在根目录下
