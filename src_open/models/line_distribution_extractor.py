import torch
import torch.nn as nn
# from einops.layers.torch import Rearrange

from .base_model import BaseModel
# from timm.models.efficientnet_builder import efficientnet_init_weights
from timm.models._efficientnet_builder import efficientnet_init_weights

class LineDistributionExtractor(BaseModel):
    default_conf = {
        'in_channel': 16,  # [16, 8],
        'out_channel': 1,
        'scales': [2, 1],
        'function_length': 8,
        'distribution_length': 9,
        'cat_fore_prob': True,
        'cat_distribution': True,

        'debug_check_display': False
    }

    def _init(self, conf):
        self.conf = conf

        # self.conv = nn.Sequential(
        #     nn.Conv2d(conf.in_channel, 2 * conf.in_channel, 3),
        #     # nn.Conv2d(3, conf.out_channel, 1),
        # )
        self.final_length = conf.distribution_length + conf.function_length - 1
        self.aggregates1 = nn.ModuleList()
        self.aggregates2 = nn.ModuleList()
        self.convs = nn.ModuleList()

        # for in_channel in conf.in_channel:
        #     for scale in conf.scales:
        #         inter_channel = scale * in_channel
        #         self.aggregates1.append(nn.Sequential(
        #             nn.Conv2d(in_channel, inter_channel * 2, kernel_size=3, padding=1),
        #             nn.Conv2d(inter_channel * 2, inter_channel, kernel_size=(3, scale), stride=(1, scale),
        #                       padding=(1, 0)),
        #             nn.ReLU()
        #         ))
        #         self.aggregates2.append(nn.Sequential(
        #             nn.Conv2d(inter_channel + 1, (inter_channel + 1) * 2, kernel_size=3, padding=1),
        #             nn.Conv2d((inter_channel + 1) * 2, inter_channel + 1,
        #                       kernel_size=(3, conf.function_length), padding=(1, 0)),
        #             # nn.InstanceNorm2d(inter_channel)
        #             # TODO: relu
        #             nn.ReLU()
        #         ))
        #         self.convs.append(nn.Sequential(
        #             nn.Conv2d(inter_channel + 2, (inter_channel + 2) * 2, kernel_size=3, padding=1),
        #             nn.Conv2d((inter_channel + 2) * 2, (inter_channel + 2) * 2, kernel_size=3, padding=1),
        #             # nn.Conv2d(inter_channel * 2, inter_channel, kernel_size=3, padding=1),
        #             nn.Conv2d((inter_channel + 2) * 2, conf.out_channel, kernel_size=1)
        #         ))

        for scale in conf.scales:
            inter_channel = scale*conf.in_channel

            self.aggregates1.append(nn.Sequential(
                nn.Conv2d(conf.in_channel, inter_channel * 2, kernel_size=3, padding=1),
                nn.Conv2d(inter_channel * 2, inter_channel, kernel_size=(3, scale), stride=(1, scale), padding=(1, 0)),
                nn.ReLU()
            ))
            self.aggregates2.append(nn.Sequential(
                nn.Conv2d(inter_channel + 1, (inter_channel + 1) * 2, kernel_size=3, padding=1),
                nn.Conv2d((inter_channel + 1) * 2, inter_channel + 1,
                          kernel_size=(3, conf.function_length), padding=(1, 0)),
                # nn.InstanceNorm2d(inter_channel)
                # TODO: relu
                nn.ReLU()
            ))
            # self.aggregates.append(nn.Sequential(
            #     nn.Conv2d(conf.in_channel, inter_channel, kernel_size=(3, scale), stride=(1, scale), padding=(1, 0)),
            #     nn.Conv2d(inter_channel, inter_channel, kernel_size=(3, conf.function_length), padding=(1, 0)),
            #     # nn.InstanceNorm2d(inter_channel)
            #     # TODO: relu
            #     # nn.ReLU()
            # ))
            # self.avgs.append(nn.AvgPool2d((1, scale)))
            self.convs.append(nn.Sequential(
                nn.Conv2d(inter_channel + 2, (inter_channel + 2) * 2, kernel_size=3, padding=1),
                nn.Conv2d((inter_channel + 2) * 2, (inter_channel + 2) * 2, kernel_size=3, padding=1),
                # nn.Conv2d(inter_channel * 2, inter_channel, kernel_size=3, padding=1),
                nn.Conv2d((inter_channel + 2) * 2, conf.out_channel, kernel_size=1)
            ))

        # self.mlp = nn.Sequential(
        #     Rearrange('b c h w-> (b h) (c w)'),
        #     # nn.LayerNorm(conf.in_channel * conf.distribution_length),
        #     nn.Linear(conf.in_channel * conf.distribution_length, conf.out_channel),
        #     Rearrange('(b h) c-> b h c', h=200)
        # )

        # efficientnet_init_weights(self.conv)
        # efficientnet_init_weights(self.mlp)
        efficientnet_init_weights(self.convs)
        efficientnet_init_weights(self.aggregates1)
        efficientnet_init_weights(self.aggregates2)

    # just for jit trace
    # def forward(self, lines_feature, pf, distributions):
    #     x = lines_feature
    #     ind = 0  # it * len(self.conf.scales) + inner_it
    #     distributions = distributions.unsqueeze(1)
    #     pf = pf.unsqueeze(1)
    #     if not self.conf.cat_distribution:
    #         distributions = torch.zeros_like(distributions).to(distributions.device)
    #     if not self.conf.cat_fore_prob:
    #         pf = torch.zeros_like(pf).to(pf.device)

    #     x = self.aggregates1[ind](x)
    #     x = torch.cat((x, pf), dim=1)
    #     x = self.aggregates2[ind](x)
    #     x = torch.cat((x, distributions), dim=1)
    #     _, C, _, _ = x.shape
    #     x = self.convs[ind](x)
    #     softmax_temp = 1. / C**.5
    #     x = x * softmax_temp
    #     # output_distributions = torch.exp(x) / torch.sum(torch.exp(x),dim=-1, keepdim=True)
    #     output_distributions = torch.softmax(x, dim=-1)
        
    #     # import ipdb
    #     # ipdb.set_trace()

    #     # return x

    #     return output_distributions.squeeze(1)

    def _forward(self, inp):
        x = inp['lines_feature']
        it = inp['it']
        inner_it = inp['inner_it']
        ind = inner_it  # it * len(self.conf.scales) + inner_it
        distributions = inp['distributions'].unsqueeze(1)
        pf = inp['pf'].unsqueeze(1)
        if not self.conf.cat_distribution:
            distributions = torch.zeros_like(distributions).to(distributions.device)
        if not self.conf.cat_fore_prob:
            pf = torch.zeros_like(pf).to(pf.device)

        x = self.aggregates1[ind](x)
        x = torch.cat((x, pf), dim=1)
        x = self.aggregates2[ind](x)
        x = torch.cat((x, distributions), dim=1)
        _, C, _, _ = x.shape
        x = self.convs[ind](x)
        softmax_temp = 1. / C**.5
        output_distributions = torch.softmax(x * softmax_temp, dim=-1)

        return output_distributions.squeeze(1)

    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError