# -*- encoding: utf-8 -*-

from layer import GAT_Layer
import torch
from utils import pdf_norm

######################################## GAT Auto_Encoder ########################################
class AE_GAT(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_GAT, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1

        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        for index in range(self.num_layer):
            self.Encoder.append(GAT_Layer(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(GAT_Layer(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x, adj):
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x, adj)
        h = x
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x, adj)
        x_hat = x      
        return h, x_hat

####################################### DHGC ########################################
class DHGC(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, pretrain_model_load_path):
        super(DHGC, self).__init__()
        self.dims_encoder = dims_encoder

        self.AE = AE_GAT(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu')) # initialization with pretrain auto_encoder
    
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T)) # (num_nodes, num_class)
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p