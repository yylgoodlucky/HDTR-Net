import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class D_ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, name, nums=3,
                 kernel_size=3, padding=1, stride=1):
        super(D_ConvBlock, self).__init__()
        self.nums = nums
        self.relu = nn.ReLU(True)
        if isinstance(name, str):
            self.name = name
        else:
            raise Exception("name should be str")
        for i in range(self.nums):
            self.add_module('conv' + self.name + "_" + str(i), nn.Conv2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride))
            self.add_module('conv' + self.name + "_" + str(i) + "_bn", nn.BatchNorm2d(outplanes))
            inplanes = outplanes

    def forward(self, x):
        net = x
        for i in range(self.nums):
            net = self._modules['conv' + self.name + "_" + str(i)](net)
            net = self._modules['conv' + self.name + "_" + str(i) + "_bn"](net)
            net = self.relu(net)
        return net

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):  # 1, 4, 256
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.dropout = nn.Dropout(0.5)

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):  # 4, inp: [b, 256, 48, 48]
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        up1 = self.dropout(up1)
        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=1)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up1size = up1.size()
        rescale_size = (up1size[2], up1size[3])
        up2 = F.upsample(low3, size=rescale_size, mode='bilinear')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN_use(nn.Module):
    def __init__(self, opt_channel):
        super(FAN_use, self).__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(opt_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        
        # Stacking part
        hg_module = 0
        self.add_module('m' + str(hg_module), HourGlass(1, 3, 256))
        self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        self.add_module('conv_last' + str(hg_module),
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                        256, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

        if hg_module < self.num_modules - 1:
            self.add_module(
                'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('al' + str(hg_module), nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0))

        # if config.load_pretrain:
        #         self.load_pretrain(config.fan_pretrain_path2)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=3, padding=1)
        # self.avgpool = nn.MaxPool2d((2, 2), 2)
        # self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        # self.fc = nn.Linear(1024, 256)

    def forward(self, x):   # [b, 6, 96, 96]
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 1)
        x = self.conv3(x)
        x = self.conv4(x)  # [6, 256, 48, 48]

        previous = x

        i = 0
        hg = self._modules['m' + str(i)](previous)

        ll = hg            # [6, 256, 48, 48]
        ll = self._modules['top_m_' + str(i)](ll)

        ll = self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll))
        tmp_out = self._modules['l' + str(i)](F.relu(ll))
        
        return tmp_out   # [6, 68, 48, 48]

    def load_pretrain(self, pretrain_path):
        check_point = torch.load(pretrain_path)
        self.load_state_dict(check_point['check_point'])


class FanFusion(nn.Module):
    def __init__(self, opt):
        super(FanFusion, self).__init__()
        self.opt = opt
        self.mask_model = FAN_use(self.opt.mask_channel)
        self.ref_model = FAN_use(self.opt.ref_channel)

        self.m_conv1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1)
        )
        
        self.m_conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1)
        )

    def forward(self, mask, mean_mask, image):
        mask_in = torch.cat((mask, mean_mask), 1)
        
        net1 = self.mask_model.forward(mask_in)
        net2 = self.ref_model.forward(image) 
        
        net1_out = self.m_conv1(net1)
        net2_out = self.m_conv1(net2)

        return net1_out, net2_out


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv1_1_new = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.deconv1_1_bn = nn.BatchNorm2d(512)
        self.convblock1 = D_ConvBlock(512, 256, "1", nums=2)
        self.convblock2 = D_ConvBlock(256, 128, "2", nums=3)
        self.convblock3 = D_ConvBlock(128, 64, "3", nums=4)
        self.conv4_1 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.conv4_1_bn = nn.BatchNorm2d(32)
        self.conv4_2 = nn.ConvTranspose2d(32, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        net = self.deconv1_1_new(x)
        net = self.relu(self.deconv1_1_bn(net))
        for i in range(3):
            net = self._modules['convblock' + str(i + 1)](net)
            net = self.upsample(net)
        net = self.conv4_1(net)
        net = self.relu(self.conv4_1_bn(net))
        net = self.conv4_2(net)
        net = self.tanh(net)
        net = (net + 1) / 2.0
        return net
    

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.encoder = FanFusion(self.opt)
        self.decoder = Decoder(self.opt)

    def forward(self, mask, mean_mask, image):
        out_net1, out_net2 = self.encoder.forward(mask, mean_mask, image)
    
        encoder_f = torch.cat((out_net1, out_net2), 1)
        out_g = self.decoder.forward(encoder_f)
        return out_g


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
    
class perceptionLoss():
    def __init__(self):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to('cuda')
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list

    def calculatePerceptionLoss(self, video_pd, video_gt):
        # features_pd = self.getfeatures(video_pd.view(video_pd.size(0)*video_pd.size(1), video_pd.size(2), video_pd.size(3), video_pd.size(4)))
        # features_gt = self.getfeatures(video_gt.view(video_gt.size(0)*video_gt.size(1), video_gt.size(2), video_gt.size(3), video_gt.size(4)))
        features_pd = self.getfeatures(video_pd)
        features_gt = self.getfeatures(video_gt)
        
        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]
        
        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss
        