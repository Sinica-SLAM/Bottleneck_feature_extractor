import os
import math
import torch
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, log_loss, gradient_penalty_loss)

from .layers_vq import ( EMAVectorQuantizer, EncodeResidualStack, DecodeResidualStack, DecodeResidualAdaINStack, Jitter)



class Trainer(object):
    def __init__(self, train_config, model_config):
        self._gamma        = train_config.get('gamma', 1)
        self._gp_weight    = train_config.get('gp_weight', 1)
        self.pre_iter      = train_config.get('pre_iter', 1000)
        self.gen_param     = train_config.get('generator_param', {
                                'per_iteration': 1,
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })
        self.disc_param    = train_config.get("discriminator_param", {
                                'per_iteration': 1,            
                                'optim_type': 'RAdam',
                                'learning_rate': 5e-5,
                                'max_grad_norm': 1,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })


        checkpoint_path    = train_config.get('checkpoint_path', '')


        # Initial Generator and Discriminator
        self.model_G = Model(model_config['Generator'])
        self.model_D = Encoder(**model_config['Discriminator'])

        print(self.model_G)
        print(self.model_D)

        # Initial Optimizer
        self.optimizer_G = RAdam( self.model_G.parameters(), 
                                  lr=self.gen_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.optimizer_D = RAdam( self.model_D.parameters(), 
                                  lr=self.disc_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_G,
                                    **self.gen_param['lr_scheduler']
                            )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_D,
                                    **self.disc_param['lr_scheduler']
                            )

        if os.path.exists(checkpoint_path):
            self.iteration = self.load_checkpoint(checkpoint_path)
        else:
            self.iteration = 0

        self.model_G.cuda().train()
        self.model_D.cuda().train()

    def step(self, input, iteration=None):
        if iteration is None:
            iteration = self.iteration

        assert self.model_G.training 
        assert self.model_D.training

        x_batch, y_batch = input
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        loss_detail_G = dict()
        loss_detail_D = dict()

        ##########################
        # Phase 1: Train the VAE #
        ##########################
        if iteration <= self.pre_iter:
            x_real, x_fake, y_idx, loss, loss_detail_G = self.model_G((x_batch, y_batch))

            self.model_G.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        ####################################
        # Phase 2: Train the discriminator #
        ####################################
        if iteration > self.pre_iter and iteration % self.disc_param['per_iteration'] == 0:
            # Train the Discriminator
            with torch.no_grad():
                x_real, x_fake, y_idx, _, _ = self.model_G((x_batch, y_batch))

            logit_real = F.nll_loss(self.model_D(x_real), y_idx)

            if isinstance(x_fake,tuple):
                logit_fake = -F.nll_loss(self.model_D(x_fake[0]), y_idx)
                logit_fake -= F.nll_loss(self.model_D(x_fake[1]), y_idx)
                logit_fake = logit_fake / 2

                
                gp_loss =  gradient_penalty_loss(x_real, x_fake[0], self.model_D)
                gp_loss += gradient_penalty_loss(x_real, x_fake[1], self.model_D)
                gp_loss = gp_loss / 2
            else:
                logit_fake = -F.nll_loss(self.model_D(x_fake), y_idx)
                gp_loss =  gradient_penalty_loss(x_real, x_fake, self.model_D)

            disc_loss = logit_real + logit_fake
            loss = disc_loss + self._gp_weight * gp_loss

            loss_detail_D['DISC loss'] = disc_loss.item()
            loss_detail_D['gradient_penalty'] = gp_loss.item()

            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.disc_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_D.parameters(),
                    self.disc_param['max_grad_norm'])
            self.optimizer_D.step()
            self.scheduler_D.step()

        ################################
        # Phase 2: Train the generator #
        ################################
        if iteration > self.pre_iter and iteration % self.gen_param['per_iteration'] == 0:
            # Train the Generator
            x_real, x_fake, y_idx, loss, loss_detail_G = self.model_G((x_batch, y_batch))

            if isinstance(x_fake,tuple):
                adv_loss =  F.nll_loss(self.model_D(x_fake[0]), y_idx)
                adv_loss += F.nll_loss(self.model_D(x_fake[1]), y_idx)
            else:
                adv_loss =  F.nll_loss(self.model_D(x_fake), y_idx)

            loss += self._gamma * adv_loss

            loss_detail_G['Total'] = loss.item()
            loss_detail_G['ADV loss'] = adv_loss.item()
            
            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        # Get loss detail
        loss_detail = dict()
        for key, val in loss_detail_G.items():
            loss_detail[key] = val
        for key, val in loss_detail_D.items():
            loss_detail[key] = val

        self.iteration = iteration + 1

        return self.iteration, loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model_G.state_dict(),
                'discriminator': self.model_D.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint_data.keys():
            self.model_G.load_state_dict(checkpoint_data['model'])
        if 'discriminator' in checkpoint_data.keys():
            self.model_D.load_state_dict(checkpoint_data['discriminator'])
        if 'optimizer_G' in checkpoint_data.keys():
            self.optimizer_G.load_state_dict(checkpoint_data['optimizer_G'])
        if 'optimizer_D' in checkpoint_data.keys():
            self.optimizer_D.load_state_dict(checkpoint_data['optimizer_D'])
        self.scheduler_G.last_epoch = checkpoint_data['iteration']
        self.scheduler_D.last_epoch = checkpoint_data['iteration'] - self.pre_iter
        return checkpoint_data['iteration']

    def adjust_learning_rate(self, optimizer, learning_rate=None):
        if learning_rate is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = Encoder(**arch['encoder'])
        self.decoder = Decoder(**arch['decoder'])

        self.quantizer = EMAVectorQuantizer( arch['z_num'], arch['z_dim'], arch['mu'], reduction='sum')
        self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=False)

        self.jitter = Jitter(probability=arch['jitter_p'])
        
        self.beta = arch['beta']
        self.y_num = arch['y_num']
        self.z_num = arch['z_num']


    def forward(self, input):
        # Preprocess
        x, y_idx = input    # ( Size( N, x_dim, nframes), Size( N, nframes))
        y = self.embeds(y_idx).transpose(1,2).contiguous()    # Size( N, y_dim, nframes)
        # Encode
        z = self.encoder(x)

        # Decode
        if self.training:
            z_vq, z_qut_loss, z_enc_loss, vq_detail = self.quantizer(z)
            z_vq = self.jitter(z_vq)

            xhat = self.decoder((z_vq, y))

            # Loss
            Batch, Dim, Time = x.shape
            mean_factor = Batch * Time

            z_qut_loss = z_qut_loss / mean_factor
            z_enc_loss = z_enc_loss / mean_factor
            
            x_loss = log_loss(xhat, x) / mean_factor

            loss = x_loss + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_enc_loss.item(),
                      'entropy': vq_detail['entropy'].item(),
                      'usage_batch': vq_detail['used_curr'].item(),
                      'usage': vq_detail['usage'].item(),
                      'diff_emb': vq_detail['dk'].item(),
                      'X like': x_loss.item()}

            y_idx = y_idx * 0
            y_idx = y_idx[:,:1].repeat(1,Time).detach()

            return x, xhat, y_idx, loss, losses
            # return x, (xhat,xhat2), y_idx, loss, losses

        else:
            z_vq = self.quantizer(z)
            xhat = self.decoder((z_vq,y))

            return xhat


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                # logging.debug(f"Weight norm is removed from {m}.")
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


    def load_state_dict(self, state_dict):
        warning_mseg =  'Embedding size mismatch for {}: '
        warning_mseg += 'copying a param with shape {} from checkpoint, '
        warning_mseg += 'resizing the param with shape {} in current model.'

        state_dict_shape, module_param_shape = state_dict['quantizer.embeddings'].shape, self.quantizer.embeddings.shape
        if state_dict_shape != module_param_shape:
            print(warning_mseg.format('model.quantizer', state_dict_shape, module_param_shape))
            self.quantizer = VectorQuantizer( 
                    state_dict_shape[0], state_dict_shape[1], 
                    normalize=self.quantizer.normalize, reduction=self.quantizer.reduction
                    )
        super(Model, self).load_state_dict(state_dict)


class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels=[513, 1024, 512, 256],
                 out_channels=[1024, 512, 256, 128],
                 downsample_scales=[1, 1, 1, 1],
                 kernel_size=5,
                 z_channels=128,
                 bias=True,
                 dilation=True,
                 stack_kernel_size=3,
                 stack_layers=2,
                 stacks=[3, 3, 3, 3],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 normalization_func="GroupNorm",
                 normalization_params={ "num_groups": 1,
                                        "eps": 1e-05, 
                                        "affine": True},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_weight_norm=True,
                 use_causal_conv=False,
                 ):
        super(Encoder, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = []

        for ( in_channel, out_channel, ds_scale, stack) in zip( in_channels, out_channels, downsample_scales, stacks):

            if ds_scale == 1:
                _kernel_size = kernel_size
                _padding = (kernel_size - 1) // 2
                _stride = 1
            else:
                _kernel_size = ds_scale * 2
                _padding = ds_scale // 2 + ds_scale % 2
                _stride = ds_scale

            layers += [
                getattr(torch.nn, pad)( _padding, **pad_params),
                torch.nn.Conv1d(in_channel, out_channel, _kernel_size, stride=_stride, bias=bias)
            ]

            # add residual stack
            for j in range(stack):
                layers += [
                    EncodeResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=out_channel,
                        layers=stack_layers,
                        dilation=2**j if dilation else 1,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        normalization_func=normalization_func,
                        normalization_params=normalization_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]

        # add final layer
        layers += [torch.nn.Conv1d( out_channels[-1], z_channels, 1, bias=bias)]

        self.encode = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            input (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.encode(input)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class Decoder(torch.nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=[256, 512, 1024, 513],
                 upsample_scales=[1, 1, 1, 1],
                 cond_channels=128,
                 skip_channels=80,
                 final_channels=80,
                 kernel_size=5,
                 bias=True,
                 dilation=True,
                 stack_kernel_size=3,
                 stacks=[3, 3, 3, 3],
                 nonlinear_activation="GLU",
                 nonlinear_activation_params={},
                 normalization_func="GroupNorm",
                 normalization_params={ "num_groups": 1,
                                        "eps": 1e-05, 
                                        "affine": True},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_adain=False,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 ):
        super(Decoder, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = torch.nn.ModuleList()

        for ( in_channel, out_channel, us_scale, stack) in zip( in_channels, out_channels, upsample_scales, stacks):
            # add resampling layer
            if us_scale == 1:
                _kernel_size = kernel_size
                padding = (kernel_size - 1) // 2
                output_padding = 0
                stride = 1
            else:
                _kernel_size = us_scale * 2
                padding = us_scale // 2 + us_scale % 2
                output_padding = us_scale % 2
                stride = us_scale

            layers += [
                torch.nn.ConvTranspose1d( 
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=_kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias)
            ]
            # add residual stack
            for j in range(stack):
                if use_adain:
                    DecodeStack = DecodeResidualAdaINStack
                else:
                    DecodeStack = DecodeResidualStack
                layers += [
                    DecodeStack(
                        kernel_size=stack_kernel_size,
                        in_channels=out_channel,
                        cond_channels=cond_channels,
                        skip_channels=skip_channels,
                        dilation=2**j if dilation else 1,
                        bias=True,
                        dropout=0.0,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        normalization_func=normalization_func,
                        normalization_params=normalization_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]             
            
        # add final layer
        final_layer = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv1d( skip_channels, skip_channels, 1, bias=bias),
                torch.nn.ReLU(),
                torch.nn.Conv1d( skip_channels, final_channels, 1, bias=bias),
            )

        self.layers = layers
        self.final_layer = final_layer

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Input tensor (B, cond_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        # return self.decode(x)
        x, c = input
        x_out = 0.0
        c = c[:,:,:1]
        for layer in self.layers:
            if isinstance(layer, DecodeResidualStack) or isinstance(layer, DecodeResidualAdaINStack):
                x, x_skip = layer( x, c.repeat(1,1,x.size(2)))
                x_out += x_skip
            else:
                x = layer(x)
        x = x_out * math.sqrt(1.0 / len(self.layers))
        x = self.final_layer(x)
        return x

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
