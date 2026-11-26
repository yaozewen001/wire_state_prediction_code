"""三个模型共用的灰度图像CNN；输出宽度随模型配置保存，不读取标签。"""

from torch import nn  # 构建共享的空间特征提取网络。


def make_image_cnn(output_dim=128):  # 新训练默认输出128维，旧模型恢复时显式传64。
    if type(output_dim) is not int or output_dim < 4 or output_dim % 4:  # 三层通道按四分之一、二分之一和完整宽度分配。
        raise ValueError("IMAGE_FEATURE_DIM必须是至少为4的4的倍数。")  # 防止非法通道数。
    first, second = output_dim // 4, output_dim // 2  # 128维配置对应32、64、128通道。
    return nn.Sequential(  # 每张历史图像与记忆新帧共享同一组参数。
        nn.Conv2d(1, first, 5, stride=2, padding=2),  # 第一层提取基础空间特征。
        nn.ReLU(),  # 引入非线性。
        nn.MaxPool2d(2),  # 降低中间图尺寸。
        nn.Conv2d(first, second, 3, stride=2, padding=1),  # 第二层提取组合特征。
        nn.ReLU(),  # 引入非线性。
        nn.Conv2d(second, output_dim, 3, stride=2, padding=1),  # 最后一层输出配置的通道数。
        nn.ReLU(),  # 引入非线性。
        nn.AdaptiveAvgPool2d(1),  # 每张图汇总为output_dim维向量。
        nn.Flatten(),  # 移除长度为1的空间维度。
    )  # 保留原Sequential索引，64维配置可恢复旧v5权重。
