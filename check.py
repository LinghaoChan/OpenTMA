import torch









if __name__ == '__main__':
    weight = torch.load('/comp_robot/lushunlin/motion-latent-diffusion/deps/t2m/motionx/version_only_humanml/smplx_212/text_mot_match_glove_6B_caption_bs_256/model/finest.tar')
    for i in weight.keys():
        for j in weight[i].keys():
            import pdb; pdb.set_trace()
            if torch.isnan(weight[i][j]).any():
                import pdb; pdb.set_trace()
                

    import pdb; pdb.set_trace()









