import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 nfeats: int,
                 mu, 
                 quantizer='ema_reset', 
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer

        dim_input = nfeats
        # if args.dataname == 'motionx':
        #     if args.motion_type == 'vector_263':
        #         dim_input = 263
        #     elif args.motion_type == 'smplx_212':
        #         dim_input = 212
        #     else:
        #         raise KeyError("no such motion type")
        # elif args.dataname == 'kit':
        #     dim_input = 251
        # elif args.dataname == 'humanml3d':
        #     dim_input = 263
        # else:
        #     raise KeyError("Dataset is not supported")
        self.encoder = Encoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(dim_input, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                 nfeats: int, 
                 mu, 
                 quantizer='ema_reset', 
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 **kwargs):
        
        super().__init__()
        # self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(nfeats, mu, quantizer, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, (commit_x, commit_x_d), perplexity = self.vqvae(x)
        
        return x_out, (commit_x, commit_x_d), perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        