from typing import List, Optional

import torch

# from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from mld.models.tools.tools import remove_padding

from mld.models.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from mld.models.modeltype.base import BaseModel
from torch.distributions.distribution import Distribution
from mld.config import instantiate_from_config

from mld.models.losses.temos import TemosLosses
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

from mld.models.architectures import t2m_textenc, t2m_motionenc
import os

import time

import numpy as np
import torch.nn.functional as f
from pathlib import Path

from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel


class TEMOS(BaseModel):
    # def __init__(self, textencoder: DictConfig,
    #              motionencoder: DictConfig,
    #              motiondecoder: DictConfig,
    #              losses: DictConfig,
    #              optim: DictConfig,
    #              transforms: DictConfig,
    #              nfeats: int,
    #              vae: bool,
    #              latent_dim: int,
    #              **kwargs):
    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.is_vae = cfg.model.vae
        self.cfg = cfg
        self.condition = cfg.model.condition
        self.stage = cfg.TRAIN.STAGE
        self.datamodule = datamodule
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.motion_type = cfg.DATASET.MOTION_TYPE

        
        self.textencoder = instantiate_from_config(cfg.textencoder)
        # import pdb; pdb.set_trace()
        self.motionencoder = instantiate_from_config(cfg.motionencoder)

        self.motiondecoder = instantiate_from_config(cfg.motiondecoder)


        # self.transforms = instantiate(transforms)
        # self.Datastruct = self.transforms.Datastruct

        # self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)

        # self.optimizer = instantiate(optim, params=self.parameters())
        # import pdb; pdb.set_trace()
        if self.condition in ["text", "text_uncond", 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
            self._get_t2m_evaluator(cfg)
            self.feats2joints = datamodule.feats2joints

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        # self._losses = torch.nn.ModuleDict({split: instantiate(losses, vae=vae,
        #                                                        _recursive_=False)
        #                                     for split in ["losses_train", "losses_test", "losses_val"]})

        self._losses = MetricCollection({
            split: TemosLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
            for split in ["losses_train", "losses_test", "losses_val"]
        })                   

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        # import pdb; pdb.set_trace()
        if self.cfg.LOSS.USE_INFONCE_FILTER:
            self.filter_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        self.retrieval_text_embedding = []
        self.retrieval_motion_embedding = []
        self.retrieval_sbert_embedding = []

        self.retrieval_corres_name = []

        self.gt_idx = 0

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:
        datastruct_from_text = self.text_to_motion_forward(batch["text"],
                                                           batch["length"])

        return remove_padding(datastruct_from_text.joints, batch["length"])


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
                                        pos_size=cfg.model.t2m_textencoder.dim_pos_ohot, # added
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
                                            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot, # added
                                            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
                                            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
                                        )
            else:
                raise NotImplementedError

        

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )


        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        
        if self.cfg.LOSS.TRAIN_TMR:
            # TMR retrival text evalutor
            vae_dict = OrderedDict()
            ckpt = "./deps/TMR-pretrained/TMR.ckpt"
            state_dict = torch.load(ckpt)["state_dict"]
            modelpath = 'distilbert-base-uncased'
            self.t2m_TMR_textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4)
            # print(model)
            for k, v in state_dict.items():
                # print(k)
                if k.split(".")[0] == "textencoder":
                    name = k.replace("textencoder.", "")
                    vae_dict[name] = v
            self.t2m_TMR_textencoder.load_state_dict(vae_dict, strict=True)
            self.t2m_TMR_textencoder.eval()
            del state_dict
            for p in self.t2m_TMR_textencoder.parameters():
                p.requires_grad = False
            
            # TMR retrival motion evalutor
            vae_dict = OrderedDict()
            ckpt = "./deps/TMR-pretrained/TMR.ckpt"
            state_dict = torch.load(ckpt)["state_dict"]
            self.t2m_TMR_motionencoder = ActorAgnosticEncoder(nfeats=263, vae =True, num_layers=4)
            # print(model)
            for k, v in state_dict.items():
                # print(k)
                if k.split(".")[0] == "motionencoder":
                    name = k.replace("motionencoder.", "")
                    vae_dict[name] = v
            self.t2m_TMR_motionencoder.load_state_dict(vae_dict, strict=True)
            self.t2m_TMR_motionencoder.eval()
            del state_dict
            for p in self.t2m_TMR_motionencoder.parameters():
                p.requires_grad = False



        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname

        if dataname == 'motionx':
            t2m_checkpoint = torch.load(
                os.path.join(cfg.model.t2m_path, dataname, cfg.DATASET.VERSION, cfg.DATASET.MOTION_TYPE, 
                            "text_mot_match/model/finest.tar"),  map_location=torch.device('cpu'))
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

        

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.is_vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return features
        return features, latent_vector, distribution

    def motion_to_motion_forward(self, features,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False,
                                 mask_ratio=0,
                                 ):
        # Make sure it is on the good device
        # datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        # mask_ratio = 0
        if self.is_vae:
            if mask_ratio<0.001:
                distribution = self.motionencoder(features, lengths)
                latent_vector = self.sample_from_distribution(distribution)
            else:
                num_mask = int(features.shape[1] * mask_ratio)
                mask = np.hstack([
                    np.zeros(num_mask, dtype=bool), np.ones(features.shape[1] - num_mask)
                ])
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = features[:, mask_shuffle, :]
                
                lengths_new = [int(mask_shuffle[:lengths[i]].sum()) for i in range(len(lengths))]
                
                distribution = self.motionencoder(masked_features, lengths_new)
                latent_vector = self.sample_from_distribution(distribution)
        else:
            if mask_ratio<0.001:
                distribution = None
                latent_vector: Tensor = self.motionencoder(features, lengths)
            else:
                num_mask = int(features.shape[1] * mask_ratio)
                mask = np.hstack([
                    np.zeros(num_mask), np.ones(features.shape[1] - num_mask)
                ])
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = features[:, mask_shuffle, :]
                distribution = None
                lengths_new = [int(mask_shuffle[:lengths[i]].sum()) for i in range(len(lengths))]
                latent_vector: Tensor = self.motionencoder(masked_features, lengths_new)
                
        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return features
        return features, latent_vector, distribution


    def save_embeddings(self, batch):
        
        with torch.no_grad():
            motion_all, text_all = None, None
            sbert_embedding_all = None
            
            texts = batch["text"]
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
            word_embs = batch["word_embs"].detach().clone()
            pos_ohot = batch["pos_ohot"].detach().clone()
            text_lengths = batch["text_len"].detach().clone()
            retrieval_name = batch['retrieval_name']

            # import pdb; pdb.set_trace()
            text_embedding = self.textencoder(texts).loc # (bs, 256)
            if self.mr < 0.001:
                motion_embedding = self.motionencoder(motions, lengths).loc # (bs, 256)
            else:
                num_mask = int(motions.shape[1] * self.mr)
                mask = np.hstack([
                    np.zeros(num_mask, dtype=bool), np.ones(motions.shape[1] - num_mask)
                ])
                np.random.shuffle(mask)
                mask_shuffle = torch.tensor(mask, dtype=torch.bool)
                masked_features = motions[:, mask_shuffle, :]
                
                lengths_new = [int(mask_shuffle[:lengths[i]].sum()) for i in range(len(lengths))]
                motion_embedding = self.motionencoder(masked_features, lengths_new).loc
            
            Emb_text = f.normalize(text_embedding, dim=1)
            Emb_motion = f.normalize(motion_embedding, dim=1)

            if text_all == None:
                text_all = Emb_text
            else:
                text_all = torch.cat((text_all, Emb_text), 0)

            if motion_all == None:
                motion_all = Emb_motion
            else:
                motion_all = torch.cat((motion_all, Emb_motion), 0)

            if self.cfg.LOSS.USE_INFONCE_FILTER:
                sbert_embedding = torch.tensor(self.filter_model.encode(texts)) # (bs, 384)
                sbert_embedding = f.normalize(sbert_embedding, dim=1)

                if sbert_embedding_all == None:
                    sbert_embedding_all = sbert_embedding
                else:
                    sbert_embedding_all = torch.cat((sbert_embedding_all, sbert_embedding), 0)
            # import pdb; pdb.set_trace()

                self.retrieval_sbert_embedding.append(sbert_embedding_all.detach().cpu().numpy())

            self.retrieval_text_embedding.append(text_all.detach().cpu().numpy())
            self.retrieval_motion_embedding.append(motion_all.detach().cpu().numpy())
            self.retrieval_corres_name.append(retrieval_name)
            
            





    def t2m_eval(self, batch):
        # import pdb; pdb.set_trace()
        retrieval_name = batch['retrieval_name']
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

        # if self.stage in ['diffusion', 'vae_diffusion']:
        #     # diffusion reverse
        #     if self.do_classifier_free_guidance:
        #         uncond_tokens = [""] * len(texts)
        #         if self.condition == 'text':
        #             uncond_tokens.extend(texts)
        #         elif self.condition == 'text_uncond':
        #             uncond_tokens.extend(uncond_tokens)
        #         texts = uncond_tokens
        #     text_emb = self.text_encoder(texts)
        #     z = self._diffusion_reverse(text_emb, lengths)
        # elif self.stage in ['vae']:
        #     if self.vae_type in ["mld", "vposert", "actor"]:
        #         z, dist_m = self.vae.encode(motions, lengths)
        #     else:
        #         raise TypeError("Not supported vae type!")
        #     if self.condition in ['text_uncond']:
        #         # uncond random sample
        #         z = torch.randn_like(z)


        assert self.stage in ['temos']

        # import pdb; pdb.set_trace()
        # Encode the text/decode to a motion 
        gt_motion = motions.clone()
        gt_lengths = lengths[:]
        gt_texts = texts[:]
        with torch.no_grad():
            rett = self.text_to_motion_forward(texts,
                                            lengths,
                                            return_latent=True)
            feat_from_text, latent_from_text, distribution_from_text = rett

            # Encode the motion/decode to a motion
            retm = self.motion_to_motion_forward(motions,
                                                lengths,
                                                return_latent=True)
            feat_from_motion, latent_from_motion, distribution_from_motion = retm
        clone_feat_from_text = feat_from_text.clone()
        # with torch.no_grad():
        #     if self.vae_type in ["mld", "vposert", "actor"]:
        #         feats_rst = self.vae.decode(z, lengths)
        #     elif self.vae_type == "no":
        #         feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feat_from_text, self.cfg.DATASET.MOTION_TYPE)
        joints_ref = self.feats2joints(motions, self.cfg.DATASET.MOTION_TYPE)

        # import pdb; pdb.set_trace()
        # for i in range(joints_ref.shape[0]):
        #     np.save(os.path.join("/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_motion_debug", '{:0{width}}.npy'.format(self.gt_idx, width=5)), joints_ref[i][:lengths[i],].detach().cpu().numpy())
        #     # np.save(os.path.join("/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_motion", '{:0{width}}.npy'.format(self.gt_idx, width=5)), joints_ref[0][:lengths[0],].detach().cpu().numpy())
        #     with open(os.path.join("/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_text_debug", '{:0{width}}.txt'.format(self.gt_idx, width=5)), "w") as test_file:
        #         test_file.write(texts[i])
        #     with open(os.path.join("/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_name_debug", '{:0{width}}.txt'.format(self.gt_idx, width=5)), "w") as test_name_file:
        #         test_name_file.write(retrieval_name[i])
        #     self.gt_idx += 1

        

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feat_from_text)
        motions = self.datamodule.renorm4t2m(motions)
        # import pdb; pdb.set_trace()
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

        TMR_motion_embedding = None
        TMR_GT_motion_embedding = None
        TMR_text_embedding = None    
        if self.cfg.LOSS.TRAIN_TMR:
            # import pdb; pdb.set_trace()
            # 可能是这里的问题
            
            # TMR_motion_embedding = self.t2m_TMR_motionencoder(feat_from_text.detach(), lengths).loc.detach()
            TMR_motion_embedding = self.t2m_TMR_motionencoder(feat_from_motion.detach(), lengths).loc.detach()
            TMR_GT_motion_embedding = self.t2m_TMR_motionencoder(gt_motion.detach(), lengths).loc.detach()
            TMR_text_embedding = self.t2m_TMR_textencoder(texts).loc.detach()
        
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

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "TMR_motion_embedding": TMR_motion_embedding,
            "TMR_GT_motion_embedding": TMR_GT_motion_embedding,
            # "TMR_GT_motion_embedding": distribution_from_text.loc,
            "TMR_text_embedding": TMR_text_embedding,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):

        emb_dist = None
        if self.cfg.LOSS.USE_INFONCE and self.cfg.LOSS.USE_INFONCE_FILTER:
            # import pdb ;pdb.set_trace()
            with torch.no_grad():
                text_embedding = self.filter_model.encode(batch["text"])
                text_embedding = torch.tensor(text_embedding).to(batch['motion'][0])
                normalized = f.normalize(text_embedding, p=2, dim=1)
                emb_dist = normalized.matmul(normalized.T)

        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        feat_from_text, latent_from_text, distribution_from_text = ret

        # Encode the motion/decode to a motion
        self.mr=0
        mr = self.mr
        if split in ["train"]:
            ret = self.motion_to_motion_forward(batch["motion"],
                                                batch["length"],
                                                return_latent=True,
                                                mask_ratio=mr)
        else:
            ret = self.motion_to_motion_forward(batch["motion"],
                                                batch["length"],
                                                return_latent=True,
                                                mask_ratio=mr)
            
        feat_from_motion, latent_from_motion, distribution_from_motion = ret
        
        sync = self.cfg.LOSS.SYNC
        TMR_motion_embedding = None
        TMR_text_embedding = None
        # if sync:
        #     sync_ret_text = self.motion_to_motion_forward(
        #                                     feat_from_text,
        #                                     batch["length"],
        #                                     return_latent=True,
        #                                     mask_ratio=0,
        #                                 )
        #     feat_from_text_sync, latent_from_text_sync, distribution_from_text_sync = sync_ret_text
        # else:
        #     feat_from_text_sync, latent_from_text_sync, distribution_from_text_sync, sync_ret_text = None, None, None, None
        if sync: 
            TMR_motion_embedding = self.t2m_TMR_motionencoder(feat_from_text, batch["length"]).loc
            # TMR_motion_embedding = self.t2m_TMR_motionencoder(feat_from_motion, batch["length"]).loc
            TMR_text_embedding = self.t2m_TMR_textencoder(batch["text"]).loc
        
        
        # import pdb; pdb.set_trace()
        # GT data
        # datastruct_ref = batch["datastruct"]

        # Compare to a Normal distribution
        if self.is_vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        # import pdb; pdb.set_trace()
        # Compute the losses
        loss = self.losses[split].update(f_text=feat_from_text,
                                         f_motion=feat_from_motion,
                                         f_ref=batch["motion"],
                                         lat_text=latent_from_text,
                                         lat_motion=latent_from_motion,
                                         dis_text=distribution_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_ref=distribution_ref, 
                                         emb_dist=emb_dist,
                                         TMR_text_embedding = TMR_text_embedding,
                                         TMR_motion_embedding = TMR_motion_embedding,
                                         )

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        
        
        if split in ["val", "test"]:
            self.save_embeddings(batch)
            
            if self.condition in ['text', 'text_uncond', 'text_all', 'text_face', 'text_body', 'text_hand', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            else:
                raise NotImplementedError

            # import pdb; pdb.set_trace()
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            # import pdb; pdb.set_trace()
            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                            "motionx",
                            "unimocap",
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
                        rs_set['lat_t'],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                        rs_set["TMR_motion_embedding"],
                        rs_set["TMR_GT_motion_embedding"],
                        rs_set["TMR_text_embedding"],
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


        if split in ["test"]:
            if self.motion_type == 'vector_263':
                # import pdb; pdb.set_trace()
                return rs_set["joints_rst"], batch["length"], batch["text"]
            elif self.motion_type == 'smplx_212':
                if self.cfg.TRAIN.use_joints:
                    # import pdb; pdb.set_trace()
                    return rs_set["m_rst"], batch["length"], rs_set["m_ref"]
                else:
                    return batch["length"]

        return loss


    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["val", "test"]:
            
            

            # import pdb; pdb.set_trace()
            if (self.trainer.current_epoch+1) % 100 == 0:
            # if True:
                # import pdb; pdb.set_trace()
                # output_dir = Path(
                #     os.path.join(
                #         self.cfg.FOLDER,
                #         str(self.cfg.model.model_type),
                #         str(self.cfg.NAME),
                #         "embeddings",
                #         split,
                #         "epoch_" + str(self.trainer.current_epoch)
                #     ))


                output_dir = Path(
                    os.path.join(
                        self.cfg.FOLDER,
                        str(self.cfg.model.model_type),
                        str(self.cfg.NAME),
                        "embeddings",
                        split,
                        "epoch_" + str(self.trainer.current_epoch)
                    ))
                
                os.makedirs(output_dir, exist_ok=True)
                
                # import pdb; pdb.set_trace()
                
                # [i.squeeze() for i in self.all_gather(self.retrieval_text_embedding)]
                # print('self.retrieval_text_embedding length: ', len(self.retrieval_text_embedding))
                # print('self.retrieval_text_embedding type: ', type(self.retrieval_text_embedding))
                # print('self.all_gather(self.retrieval_text_embedding) length', len(self.all_gather(self.retrieval_text_embedding)))
                # print('self.all_gather(self.retrieval_text_embedding) type', type(self.all_gather(self.retrieval_text_embedding)))
                # print(self.all_gather(self.retrieval_text_embedding)[0].shape)
                # print('++++++++++++++')
                # print('self.retrieval_text_embedding hou shape', torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_text_embedding)], dim=0).shape)
                self.retrieval_text_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_text_embedding)], dim=0)
                self.retrieval_motion_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_motion_embedding)], dim=0)
                

                tmp_retrieval_name = []
                for i in self.all_gather(self.retrieval_corres_name):
                    tmp_retrieval_name += i
                self.retrieval_corres_name = tmp_retrieval_name
                with open(output_dir/"test_name_debug.txt", "w") as test_name_file:
                    for i in self.retrieval_corres_name:
                        test_name_file.write(i + '\n')
                
                # self.retrieval_corres_name = [tmp_retrieval_name + i for i in self.all_gather(self.retrieval_corres_name)]

                if self.cfg.LOSS.USE_INFONCE_FILTER:
                    self.retrieval_sbert_embedding = torch.cat([i.view(-1, i.shape[-1]) for i in self.all_gather(self.retrieval_sbert_embedding)], dim=0)
                    np.save(output_dir/"sbert_embedding.npy", self.retrieval_sbert_embedding.detach().cpu().numpy())


                
                np.save(output_dir/"text_embedding.npy", self.retrieval_text_embedding.detach().cpu().numpy())# (2324, 256)
                np.save(output_dir/"motion_embedding.npy", self.retrieval_motion_embedding.detach().cpu().numpy())

                print('save embedding in {} at {}'.format(output_dir, self.trainer.current_epoch))
                
            # import pdb; pdb.set_trace()
            self.retrieval_text_embedding = []
            self.retrieval_motion_embedding = []
            self.retrieval_sbert_embedding = []

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val", "test"]:

            if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(
                    self,
                    metric).compute(sanity_flag=self.trainer.sanity_checking)
                # reset metrics
                getattr(self, metric).reset()
                dico.update({
                    f"Metrics/{metric}": value.item()
                    for metric, value in metrics_dict.items()
                })
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        return self.allsplit_epoch_end("train", outputs)
