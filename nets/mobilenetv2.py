from torch import nn
from torch.hub import load_state_dict_from_url



def _make_divisible(v,divisor,min_value=None):

    """
    # 确保所有的通道数被8整除
    :param v:  # 通道数
    :param divisor:  除数 8
    :param min_value:  最小除数
    """
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value,int(v+divisor/2)//divisor * divisor)

    if new_v<0.9*v: # 新的通道数小于原来的90% 则加divisor(8)
        new_v+=divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    # 常规的conv+bn+relu ，默认卷积核大小为3，步长为1
    def __init__(self,in_channels,out_channels,kernel_size=3, stride=1, groups=1):
        padding =(kernel_size-1)//2 #取整
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.out_channels=out_channels

class InvertedResidual(nn.Module):
    #残差块
    def __init__(self,in_channels,out_channels,stride, expand_ratio):
        #  输入通道数，输出通道数，步长，扩展率
        super(InvertedResidual, self).__init__()
        self.stride=stride
        assert stride in [1,2]
        
        hidden_dim=int(round(in_channels*expand_ratio)) # 中间隐藏层的通道数
        
         # 如果步长为1 and 输入通道数与输出通道数相等，则用残差模块
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            # 如果扩展率不为1，则用扩张，扩张输出通道数为expand_ratio（6）倍，即添加卷积核大小为1x1的卷积操作
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        
        layers.extend([
            # dw 输入与输出层相等，步长等于设置参数，即特征提取
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear压缩层， 即线性层操作
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
    def forward(self,x):
        if self.use_res_connect: #残差操作
            return x+self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block=InvertedResidual
        input_channel = 32 #第一层输入通道
        last_channel = 1280 #最后一层输出通道

        if inverted_residual_setting is None:
            # t, c, n, s  t表示扩张因子，c表示输出的通道数，n表示该Bottleneck重复的次数，s表示卷积的步长
            inverted_residual_setting=[
                [1, 16, 1, 1], # 不扩张， 输出16，重复1次，步长1
                [6, 24, 2, 2], # 扩张倍数6倍， 输出24，重复2次，步长2
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],# 扩张倍数6倍 ，输出160，重复3次，步长2
                [6, 320, 1, 1],
            ]
        
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
    
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]  # 第一层，步长为2
        for t,c,n,s in inverted_residual_setting:
            output_channel=_make_divisible(c*width_mult,round_nearest)
            for i in range(n):
                stride=s if i==0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 最后一层特征提取层 （320,1280,1）
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        
        # 建立分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),#dropout层
            nn.Linear(self.last_channel, num_classes),# 线性层 （1280.1000）
        )
        
        #权重初始化
        for m in self.modules():# modules类型是OrderedDict()字典
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # w用kaiming初始化
                if m.bias is not None:# b用0初始化
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d): # BN层初始化
                nn.init.ones_(m.weight)   # w用1
                nn.init.zeros_(m.bias) # b用0
            elif isinstance(m, nn.Linear):# 线性层初始化
                nn.init.normal_(m.weight, 0, 0.01) # w用normal_
                nn.init.zeros_(m.bias) #b用0
    def forward(self, x):
        x = self.features(x) # torch.Size([1,3,244,244]) -> torch.Size([1, 1280, 7, 7])
        x = x.mean([2, 3]) # torch.Size([1, 1280]) 对h,w的均值，即求每个通道的值的平均值
        x = self.classifier(x) # torch.Size([1, 1000])
        return x
    
def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', model_dir="./model_data", progress=progress)
        model.load_state_dict(state_dict)
    del model.classifier
    return model

if __name__ == "__main__":
    net = mobilenet_v2()
    for i, layer in enumerate(net.features):
        print(i, layer)