import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses, MLDLosses_no_joint
from mld.models.losses.vqvae import VQVAELosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding

from .base import BaseModel
from .smplx_layer import smplx_layer

from ..body_skeleton.skeleton import Skeleton
from ..body_skeleton.paramUtil import *


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule
        self.motion_type = cfg.DATASET.MOTION_TYPE
        # 
        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        self.smplx_model = smplx_layer()

        self.smplx_model.eval()
        for p in self.smplx_model.parameters():
            p.requires_grad = False


        if self.vae_type != "no":
            # import pdb; pdb.set_trace()
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert", "actor", "humanvq"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        # import pdb; pdb.set_trace()
        # if self.cfg.TRAIN.STAGE not in ["vae"]:
        if cfg.DATASET.MOTION_TYPE in ['vector_263', 'smplx_212']:
            if self.condition in ["text", "text_uncond", 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            # assert cfg.DATASET.MOTION_TYPE in ['vector_263', 'root_position']
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        elif cfg.LOSS.TYPE == "vqvae":

            self._losses = MetricCollection({
                split: VQVAELosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        elif cfg.LOSS.TYPE == 'mld_no_joint':
            assert 'smpl' not in cfg.DATASET.MOTION_TYPE
            self._losses = MetricCollection({
                split: MLDLosses_no_joint(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })

        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")


        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time

        if eval("self.cfg.TRAIN.DATASETS")[0].lower() == 'humanml3d':
            n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
            kinematic_chain = t2m_kinematic_chain
        elif eval("self.cfg.TRAIN.DATASETS")[0].lower() == 'kit':
            n_raw_offsets = torch.from_numpy(kit_raw_offsets)
            kinematic_chain = kit_kinematic_chain
        else:
            raise NotImplementedError


        self.skel=None
        if self.motion_type == 'root_rot6d':
            example_data = np.load(os.path.join('/comp_robot/lushunlin/HumanML3D-1/joints', '000021' + '.npy'))
            example_data = example_data.reshape(len(example_data), -1, 3)
            example_data = torch.from_numpy(example_data)
            tgt_skel = Skeleton(n_raw_offsets, kinematic_chain)
            # (joints_num, 3)
            tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
            self.skel = Skeleton(n_raw_offsets, kinematic_chain)
            self.skel.set_offset(tgt_offsets)
            # import pdb; pdb.set_trace()

        
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond', "text_all", 'text_body', 'text_hand', 'text_face_body', 'text_face', "text_seperate", "only_pose_concat", "only_pose_fusion"]:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """

        
        # init module
        if cfg.model.eval_text_source == 'token':

            self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(word_size=cfg.model.t2m_textencoder.dim_word,
                                        pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
                                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                       )
        elif cfg.model.eval_text_source == 'only_text_token':

            self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(word_size=cfg.model.t2m_textencoder.dim_word,
                                        hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                        output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                       )

        elif cfg.model.eval_text_source in ['caption']:


            if cfg.model.eval_text_encode_way == 'clip':
                self.t2m_textencoder, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
                clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
                self.t2m_textencoder.eval()
                for p in text_enc.parameters():
                    p.requires_grad = False

            elif cfg.model.eval_text_encode_way == 't5':
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.t2m_textencoder = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
                self.t2m_textencoder.eval()
                for p in self.t2m_textencoder.parameters():
                    p.requires_grad = False

            elif 'GRU' in cfg.model.eval_text_encode_way:
                self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCoV2(word_size=cfg.model.t2m_textencoder.dim_word,
                                            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                        )
            else:
                raise NotImplementedError

        

        if cfg.DATASET.MOTION_TYPE == 'vector_263':
            self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
                input_size=cfg.DATASET.NFEATS - 4,
                hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
                output_size=cfg.model.t2m_motionencoder.dim_move_latent,
            )
        elif cfg.DATASET.MOTION_TYPE == 'smplx_212':
            self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
                input_size=cfg.DATASET.NFEATS,
                hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
                output_size=cfg.model.t2m_motionencoder.dim_move_latent,
            )
        else:
            raise NotImplementedError

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )

        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        # t2m_checkpoint = torch.load(
        #     os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, 
        #                  "text_mot_match_glove_6B_caption_bs_256/model/finest.tar"))


        if dataname == 'motionx':
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, 
                            "text_mot_match_glove_6B_caption_bs_256/model/finest.tar"),  map_location=torch.device('cpu'))
        else:
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname,
                            "text_mot_match/model/finest.tar"),  map_location=torch.device('cpu'))

        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])


        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        import pdb; pdb.set_trace()
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        elif self.vae_type in ["humanvq"]:
            feats_rst, (commit_x, commit_x_d), perplexity = self.vae.forward(feats_ref)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        if self.vae_type in ["mld", "vposert", "actor"]:
            recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition in ["text", "text_all", 'text_hand', 'text_body', 'text_face', "text_seperate", "only_pose_concat", "only_pose_fusion"]:

            if self.motion_type in ['vector_263', 'root_position', 'root_position_vel', 'root_position_rot6d']:
                joints_rst = self.feats2joints(feats_rst, self.motion_type) # feats_rst.shape (bs, seq, 67) joints_rst.shape (bs, seq, 22, 3)
                joints_ref = self.feats2joints(feats_ref, self.motion_type)
            elif self.motion_type in ['root_rot6d']:
                joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.motion_type)
                joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
                joints_ref = self.feats2joints(feats_ref, skel=self.skel, motion_type=self.motion_type)
                joints_ref = joints_ref.view(feats_ref.shape[0], feats_ref.shape[1], self.njoints, 3)
                # import pdb; pdb.set_trace()
            elif self.motion_type == 'smplx_212' and self.cfg.TRAIN.use_joints:
                joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
                joints_ref = self.feats2joints(feats_ref, self.motion_type, self.smplx_model)
            else:
                raise NotImplementedError

        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if self.vae_type in ["mld", "vposert", "actor"]:
            if dist_m is not None:
                if self.is_vae:
                    # Create a centred normal distribution to compare with
                    mu_ref = torch.zeros_like(dist_m.loc)
                    scale_ref = torch.ones_like(dist_m.scale)
                    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
                else:
                    dist_ref = dist_m

        # import pdb; pdb.set_trace()
        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        # import pdb; pdb.set_trace()
        if self.vae_type in ["humanvq"]:

            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                "commit_x": commit_x,
                "commit_x_d": commit_x_d
            }

            return rs_set
        
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                "lat_m": motion_z.permute(1, 0, 2),
                "lat_rm": recons_z.permute(1, 0, 2),
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
                "dist_m": dist_m,
                "dist_ref": dist_ref,
            }
        else:

            rs_set = {
                "m_ref": feats_ref[:, :min_len, :],
                "m_rst": feats_rst[:, :min_len, :],
                # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
                "lat_m": motion_z.permute(1, 0, 2),
                "lat_rm": recons_z.permute(1, 0, 2),
                "dist_m": dist_m,
                "dist_ref": dist_ref,
            }

        if self.cfg.LOSS.Velocity_loss:
            vel_ref = feats_ref[:, :min_len, :][:, 1:, 3:] - feats_ref[:, :min_len, :][:, :-1, 3:]
            vel_rst = feats_rst[:, :min_len, :][:, 1:, 3:] - feats_rst[:, :min_len, :][:, :-1, 3:]
            rs_set['vel_rst'] = vel_rst
            rs_set['vel_ref'] = vel_ref


        return rs_set

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist = self.vae.encode(feats_ref, lengths)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_all"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_face"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_body"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['body_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ["text_hand"]:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['hand_text'][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ['text_face_body']:
            text = []

            for i in range(len(batch["text"])):
                text.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ["text_seperate"]:
            
            text = []
            for i in range(len(batch["text"])):
                text.append((batch["text"][i], batch["face_text"][i], batch["body_text"][i], batch["hand_text"][i]))
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                ("", "", "", "") if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            
            semantic_text = []
            face_text = []
            body_text = []
            hand_text = []
            for i in range(len(text)):
                semantic_text.append(text[i][0])
                face_text.append(text[i][1])
                body_text.append(text[i][2])
                hand_text.append(text[i][3])

            cond_emb_semantic = self.text_encoder(semantic_text)
            cond_emb_face = self.text_encoder(face_text)
            cond_emb_body = self.text_encoder(body_text)
            cond_emb_hand = self.text_encoder(hand_text)
            # import pdb; pdb.set_trace()
            cond_emb = self.linear_fusion(cond_emb_semantic, cond_emb_face, cond_emb_body, cond_emb_hand)

        elif self.condition in ["only_pose_concat"]:
            text = []
            for i in range(len(batch["text"])):
                text.append(batch["face_text"][i] +' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)

        elif self.condition in ["only_pose_fusion"]:

            text = []
            for i in range(len(batch["text"])):
                text.append((batch["face_text"][i], batch["body_text"][i], batch["hand_text"][i]))
            # import pdb; pdb.set_trace()
            # text = batch["text"] +' ' + batch["body_text"] + ' ' + batch["hand_text"]
            # classifier free guidance: randomly drop text during training
            text = [
                ("", "", "") if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            
            face_text = []
            body_text = []
            hand_text = []
            for i in range(len(text)):
                face_text.append(text[i][0])
                body_text.append(text[i][1])
                hand_text.append(text[i][2])

            cond_emb_face = self.text_encoder(face_text)
            cond_emb_body = self.text_encoder(body_text)
            cond_emb_hand = self.text_encoder(hand_text)


            cond_emb = self.linear_fusion(None,cond_emb_face, cond_emb_body, cond_emb_hand)
            # emb_cat = torch.cat((cond_emb_face, cond_emb_body), axis=1)
            # emb_cat = emb_cat.view(emb_cat.size(0), -1)
            # cond_emb = self.emb_fuse(emb_cat).unsqueeze(1)

        
        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        import pdb; pdb.set_trace()
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    def t2m_eval(self, batch):
        # import pdb; pdb.set_trace()
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq"]:
                quants = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst, self.motion_type)
        joints_ref = self.feats2joints(motions, self.motion_type)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError

        
        # if self.cfg.LOSS.Velocity_loss:
        #     import pdb; pdb.set_trace()
        #     vel_ref = motions
        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set
    


    def normal_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq"]:
                quants = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq"]:
                feats_rst = self.vae.forward_decoder(quants)
                feats_rst = feats_rst.reshape(motions.shape[0], motions.shape[1], -1)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        # joints_rst = self.feats2joints(feats_rst, self.motion_type)
        # joints_ref = self.feats2joints(motions, self.motion_type)

        joints_rst = self.feats2joints(feats_rst, skel=self.skel, motion_type=self.motion_type)
        joints_rst = joints_rst.view(feats_rst.shape[0], feats_rst.shape[1], self.njoints, 3)
        joints_ref = self.feats2joints(motions, skel=self.skel, motion_type=self.motion_type)
        joints_ref = joints_ref.view(motions.shape[0], motions.shape[1], self.njoints, 3)


        rs_set = {
            "m_ref": motions,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set


    def t2m_eval_smplx(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()
        # import pdb; pdb.set_trace()
        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            elif self.vae_type in ["humanvq"]:
                quants = self.vae.encode(motions)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type in ["humanvq"]:
                feats_rst = self.vae.decode(quants)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("Not supported vae type!")

        # end time
        end = time.time()
        self.times.append(end - start)
        # import pdb; pdb.set_trace()
        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set





    def t2m_eval_smplx_text_all(self, batch):
        assert self.condition == 'text_all'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i] + ' ' + batch["hand_text"][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_all':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        import pdb; pdb.set_trace()

        return rs_set



    def t2m_eval_smplx_text_face(self, batch):
        assert self.condition == 'text_face'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_face':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set





    def t2m_eval_smplx_text_body(self, batch):
        assert self.condition == 'text_body'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['body_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_body':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set




    def t2m_eval_smplx_text_hand(self, batch):
        assert self.condition == 'text_hand'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['hand_text'][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_hand':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set



    def t2m_eval_smplx_text_face_body(self, batch):
        assert self.condition == 'text_face_body'
        texts = []
        for i in range(len(batch["text"])):
            texts.append(batch["text"][i] +' ' + batch['face_text'][i] + ' ' + batch["body_text"][i]) 

        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text_face_body':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                else:
                    raise NotImplementedError
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        if self.cfg.TRAIN.use_joints:
            joints_rst = self.feats2joints(feats_rst, self.motion_type, self.smplx_model)
            joints_ref = self.feats2joints(motions, self.motion_type, self.smplx_model)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)
        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        assert self.motion_type == 'smplx_212'

        # for p in self.t2m_moveencoder.parameters():
        #     if torch.isnan(p).any():
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        recons_mov = self.t2m_moveencoder(feats_rst).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        if self.cfg.model.eval_text_source == 'token':
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source == 'only_text_token':
            text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
        elif self.cfg.model.eval_text_source in ['caption']:
            if self.cfg.model.eval_text_encode_way == 'clip':
                raise NotImplementedError

            elif self.cfg.model.eval_text_encode_way == 't5':
                raise NotImplementedError

            elif 'GRU' in self.cfg.model.eval_text_encode_way:
                text_emb = self.t2m_textencoder(word_embs, text_lengths)[align_idx]
            else:
                raise NotImplementedError
        if self.cfg.TRAIN.use_joints:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
                "joints_ref": joints_ref,
                "joints_rst": joints_rst,
            }
        else:
            rs_set = {
                "m_ref": motions,
                "m_rst": feats_rst,
                "lat_t": text_emb,
                "lat_m": motion_emb,
                "lat_rm": recons_emb,
            }
        # import pdb; pdb.set_trace()

        return rs_set



    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        import pdb; pdb.set_trace()
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # import pdb; pdb.set_trace()
        if split in ["train", "val"]:
            if self.stage == "vae":
                if self.vae_type in ["mld", "vposert","actor"]:
                    rs_set = self.train_vae_forward(batch)
                    rs_set["lat_t"] = rs_set["lat_m"]
                else:
                    rs_set = self.train_vae_forward(batch)
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            
            else:
                raise ValueError(f"Not support this stage {self.stage}!")
            # import pdb; pdb.set_trace()
            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # import pdb; pdb.set_trace()
            if self.condition in ['text', 'text_uncond', 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                # use t2m evaluators
                if self.motion_type == 'vector_263':
                    if self.condition == 'text':
                        rs_set = self.t2m_eval(batch)
                    else:
                        raise NotImplementedError
                elif self.motion_type == 'smplx_212':
                    if self.condition == 'text':
                        rs_set = self.t2m_eval_smplx(batch)
                    elif self.condition == 'text_all':
                        rs_set = self.t2m_eval_smplx_text_all(batch)
                    elif self.condition == 'text_face':
                        rs_set = self.t2m_eval_smplx_text_face(batch)
                    elif self.condition == 'text_body':
                        rs_set = self.t2m_eval_smplx_text_body(batch)
                    elif self.condition == 'text_hand':
                        rs_set = self.t2m_eval_smplx_text_hand(batch)
                    elif self.condition == 'text_face_body':
                        rs_set = self.t2m_eval_smplx_text_face_body(batch)
                    else:
                        raise NotImplementedError
                elif self.motion_type in ['root_position', 'root_position_vel', 'root_position_rot6d', 'root_rot6d']:
                    rs_set = self.normal_eval(batch)
                else:
                    raise NotImplementedError
            
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            else:
                raise NotImplementedError
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                            "motionx"
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        # self.datamodule.renorm4t2m
        if split in ["test"]:
            if self.motion_type == 'vector_263':
                return rs_set["joints_rst"], batch["length"]
            elif self.motion_type == 'smplx_212':
                if self.cfg.TRAIN.use_joints:
                    # import pdb; pdb.set_trace()
                    return rs_set["m_rst"], batch["length"], rs_set["m_ref"]
                else:
                    return batch["length"]
        return loss
