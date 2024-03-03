import os
# import cv2
import shutil
from PIL import Image
# for cate in os.listdir('/data/huangcb/self_condition/data/extend_'):
    # if cate not in os.listdir('/data/liuwx/data/front-100/attacker_'):
    #     os.makedirs('/data/liuwx/data/front-100/defender_/'+cate,exist_ok=True)
    #     os.makedirs('/data/liuwx/data/front-100/attacker_/'+cate,exist_ok=True)
    #     os.makedirs('/data/liuwx/data/front-100/extend_/'+cate,exist_ok=True)
    # if cate not in os.listdir('/data/huangcb/self_condition/data/extend_'):
    # for img_name in os.listdir('/data/huangcb/self_condition/data/extend_/'+cate):
    #     if img_name[:9]!=cate:
            # continue
            # os.makedirs('/data/huangcb/self_condition/data/extend_/'+img_name[:9],exist_ok=True)
            # shutil.move(os.path.join('/data/huangcb/self_condition/data/extend_/'+cate,img_name),os.path.join('/data/huangcb/self_condition/data/extend_/'+img_name[:9],img_name))
    # if len(os.listdir('/data/huangcb/self_condition/data/extend_/'+cate))!=1000:
        # continue
    # os.makedirs('/data/huangcb/self_condition/data/defender/'+cate,exist_ok=True)
        # continue
        # shutil.move(os.path.join('/data/huangcb/self_condition/data/extend_/'+cate),os.path.join('/data/huangcb/self_condition/data/defender/'+cate))
    # for i,img_name in enumerate(os.listdir(os.path.join('/data/huangcb/self_condition/data/extend_',cate),)):
    #     if i<250:
            # image = Image.open(os.path.join('/data/liuwx/data/tiny-extend/tiny',cate,img_name))
        #     # resized = image.resize((hash_size + 1, hash_size), Image.ANTIALIAS)
        #     shutil.copy(os.path.join('/data/huangcb/self_condition/data/extend_',cate,img_name),os.path.join('/data/huangcb/self_condition/data/defender',cate,img_name))
            # image=cv2.imread(os.path.join('/data/liuwx/data/imagenet/res',cate,img_name))
            # resized=cv2.resize(image,(64,64), interpolation=cv2.INTER_CUBIC)
            # if i<250:
            #     cv2.imwrite(os.path.join('/data/liuwx/data/front-100/defender_/'+cate+'/'+img_name),resized)
            # elif 250<=i<500:
            #     cv2.imwrite(os.path.join('/data/liuwx/data/front-100/attacker_/'+cate+'/'+img_name),resized)
            #     continue
            # elif 500<=i<550:
            # cv2.imwrite(os.path.join('/data/liuwx/data/front-100/attacker__/'+cate+'/'+img_name),resized)

# for cate in os.listdir('/data/huangcb/self_condition/data/extend_'):
#
#     for img_name in os.listdir('/data/huangcb/self_condition/data/extend_/'+cate):
#             if img_name[:9]!=cate:
#                 os.makedirs('/data/huangcb/self_condition/data/extend_/'+img_name[:9],exist_ok=True)
#                 shutil.move(os.path.join('/data/huangcb/self_condition/data/extend_/'+cate,img_name),os.path.join('/data/huangcb/self_condition/data/extend_/'+img_name[:9],img_name))
#
for cate in os.listdir('/data/huangcb/self_condition/data/tiny_imagenet/defender/'):
    for i in range(250):
        if os.path.join(cate+f"_{i}.JPEG") not in os.listdir('/data/huangcb/self_condition/data/tiny_imagenet/defender/'+cate):
            continue
    # if len(os.listdir('/data/huangcb/self_condition/data/tiny_imagenet/defender/'+cate))!=250:
    #     continue

# for cate in os.listdir('/data/huangcb/self_condition/data/val__'):
#     if cate not in os.listdir('/data/huangcb/self_condition/data/extend_'):
#         os.makedirs('/data/huangcb/self_condition/data/extend_/'+cate,exist_ok=True)
#         os.makedirs('/data/huangcb/self_condition/data/defender/'+cate,exist_ok=True)
#         print(cate)

# for cate in os.listdir('/data/huangcb/self_condition/data/tiny_imagenet/val/front_100'):
#     os.makedirs('/data/huangcb/self_condition/data/defender/'+cate,exist_ok=True)
#     for i,img_name in enumerate(os.listdir(os.path.join('/data/huangcb/self_condition/data/extend_',cate),)):
#         if i<250:
#             shutil.copy(os.path.join('/data/huangcb/self_condition/data/extend_',cate,img_name),os.path.join('/data/huangcb/self_condition/data/defender',cate,img_name))
