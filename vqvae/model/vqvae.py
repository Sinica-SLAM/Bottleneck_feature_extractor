import math

import torch
import torch.nn.functional as F

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU, log_loss)
from .layers_vq import ( VectorQuantizer, Jitter)


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = Encoder(arch['encoder'], arch['z_dim'])
        self.decoder = Decoder(arch['decoder'], arch['y_dim'])

        self.quantizer = VectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.jitter = Jitter(probability=arch['jitter_p'])

        self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])
        
        self.beta = arch['beta']

    def forward(self, input):
        # Preprocess
        x, y = input    # ( Size( N, x_dim, nframes), Size( N, 1))
        y = self.embeds(y).transpose(1,2).contiguous()    # Size( N, y_dim, 1)

        # Encode
        z = self.encoder(x)

        # Decode
        if self.training:
            z_vq, z_qut_loss, z_enc_loss, entropy = self.quantizer(z)
            z_vq = self.jitter(z_vq)
            xhat = self.decoder((z_vq, y))

            # Loss
            Batch, Dim, Time = x.size()
            mean_factor = Batch * Time

            z_qut_loss = z_qut_loss / mean_factor
            z_enc_loss = z_enc_loss / mean_factor
            
            x_loss = log_loss(xhat, x) / mean_factor

            loss = x_loss + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_enc_loss.item(),
                      'Entropy': entropy.item(),
                      'X like': x_loss.item()}

            return loss, losses

        else:
            z_vq = self.quantizer(z)
            xhat = self.decoder((z_vq,y))

            return xhat


    def load_state_dict(self, state_dict):
        enc_state_dict = dict()
        dec_state_dict = dict()
        qtz_state_dict = dict()
        emb_state_dict = dict()
        warning_mseg =  'Embedding size mismatch for {}: '
        warning_mseg += 'copying a param with shape {} from checkpoint, '
        warning_mseg += 'resizing the param with shape {} in current model.'

        for key in state_dict.keys():
            keys = key.split('.')
            module = keys[0]
            key_new = '.'.join(keys[1:])
            if module == 'encoder':
                enc_state_dict[key_new] = state_dict[key]
            elif module == 'decoder':
                dec_state_dict[key_new] = state_dict[key]
            elif module == 'quantizer':
                state_dict_shape, module_param_shape = state_dict[key].shape, self.quantizer._embedding.shape
                if state_dict_shape != module_param_shape:
                    print(warning_mseg.format('model.quantizer', state_dict_shape, module_param_shape))
                    self.quantizer = VectorQuantizer( 
                            state_dict_shape[0], state_dict_shape[1], 
                            normalize=self.quantizer.normalize, reduction=self.quantizer.reduction
                        )
                qtz_state_dict[key_new] = state_dict[key]
            elif module == 'embeds':
                state_dict_shape, module_param_shape = state_dict[key].shape, self.embeds._embedding.weight.shape
                if state_dict_shape != module_param_shape:
                    print(warning_mseg.format('model.embeds', state_dict_shape, module_param_shape))           
                    self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=True)
                emb_state_dict[key_new] = state_dict[key]

        self.encoder.load_state_dict(enc_state_dict)
        self.decoder.load_state_dict(dec_state_dict)
        self.quantizer.load_state_dict(qtz_state_dict)
        self.embeds.load_state_dict(emb_state_dict)
        

class Encoder(torch.nn.Module):
    def __init__(self, arch, z_dim):
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            self.layers.append(
                Conv1d_Layernorm_LRelu( i, o, k, stride=s)
            )

        self.proj = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                    out_channels=z_dim,
                                    kernel_size=1)


    def forward(self, input):
        x = input   # Size( N, x_dim, nframes)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        z = self.proj(x)

        return z   # Size( N, z_dim, nframes)


class Decoder(torch.nn.Module):
    def __init__(self, arch, y_dim):
        super(Decoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            if len(self.layers) == len(arch['output']) - 1:
                self.layers.append(
                    torch.nn.ConvTranspose1d( in_channels=i+y_dim,
                                              out_channels=o,
                                              kernel_size=k,
                                              stride=s,
                                              padding=int((k-1)/2))
                )                
            else:
                self.layers.append(
                    DeConv1d_Layernorm_GLU( i+y_dim, o, k, stride=s)
                )

    def forward(self, input):
        x, y = input   # ( Size( N, z_dim, nframes), Size( N, y_dim, nframes))
        y = y.repeat(1,1,x.size(2))
        for i in range(len(self.layers)):
            x = torch.cat((x,y),dim=1)
            x = self.layers[i](x)

        return x   # Size( N, x_dim, nframes)