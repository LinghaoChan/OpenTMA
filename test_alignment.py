import os

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

render_text_folder_path = '/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_name_debug'
render_text_path_list = findAllFile(render_text_folder_path)

retrieval_f = open('/comp_robot/lushunlin/motion-latent-diffusion/results/temos/temos_humanml3d_kl_1e-5_wlatent_infonce_4gpu_nce_1e-1/embeddings/test/epoch_0/test_name_debug.txt')
retrieval_name = [i.strip() for i in retrieval_f.readlines()]

bad_count = 0
# for idx, i in enumerate(render_text_path_list):
#     render_f = open(i, 'r')
#     render_name = [j.strip() for j in render_f.readlines()]
#     if render_name != retrieval_name[idx]:
#         bad_count += 1
#         # print(render_name, retrieval_name[idx])

# 
for i in range(len(retrieval_name)):
    render_f = open(os.path.join(render_text_folder_path, '{:0{width}}.txt'.format(i, width=5)), 'r')
    render_name = render_f.readlines()[0]
    if render_name != retrieval_name[i]:
        bad_count += 1
        print(render_name, retrieval_name[idx])
    
print(bad_count)
    

