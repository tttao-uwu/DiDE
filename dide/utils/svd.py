import torch


def svd_extract_texture(feature):


    # =========================
    # remove batch
    # =========================
    F = feature.squeeze(0)      # [1374, 1024]
    F = F.float()               # 建议使用 float32

    # =========================
    # 3. 奇异值分解
    # =========================
    # F = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)

    # print("U shape :", U.shape)    # [1374, 1024]
    # print("S shape :", S.shape)    # [1024]
    # print("Vh shape:", Vh.shape)   # [1024, 1024]

    # =========================
    # 4. 低秩近似（k = 50）
    # =========================
    k = 50

    Uk = U[:, :k]          # [1374, 50]
    Sk = S[:k]             # [50]
    Vk = Vh[:k, :]         # [50, 1024]

    # 重构
    F_k = Uk @ torch.diag(Sk) @ Vk   # [1374, 1024]
    
    return F_k.unsqueeze(0) 




def svd_remove_color(feature):
    # =================================
    # 1. 输入特征
    # =================================
    # 假设 F shape = [1396, 1024]
    F = feature.squeeze(0)      # [1374, 1024]
    F = F.float()               # 建议使用 float32

    # =================================
    # 2. 参数
    # =================================
    k = 2   # 去掉前 k 个主模态（颜色 / 光照）

    # =================================
    # 3. SVD
    # =================================
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)

    # =================================
    # 4. 重构低秩部分（前 k 个模态）
    # =================================
    # 低秩 = 颜色 / 光照
    low_rank = (U[:, :k] * S[:k]) @ Vh[:k, :]

    # =================================
    # 5. 纹理部分
    # =================================
    texture_F = F - low_rank   # 高频 / 纹理部分

    # =================================
    # 6. 可选：归一化显示
    # =================================
    texture_F = (texture_F - texture_F.min()) / (texture_F.max() - texture_F.min())

    return texture_F.unsqueeze(0) 


# def svd_remove_color_sadis(feature, feature_gray, k_color=6, alpha=0.1, rank_ratio=0.5):
#     # =================================
#     # 1. 输入特征
#     # =================================
#     # 假设 F shape = [1396, 1024]
#     F_norm = feature.squeeze(0)      # [1374, 1024]
#     F_norm = F_norm.float()               # 建议使用 float32
#     Num, Dim = F_norm.shape[0], F_norm.shape[1]
    
#     F_gray = feature_gray.squeeze(0)      # [1374, 1024]
#     F_gray = F_gray.float()               # 建议使用 float32

#     fea = torch.cat([F_norm, F_gray], dim=0)
#     k_s = int(Dim * rank_ratio)
#     # 3. SVD
#     # =================================
#     Uc, Sc, Vc = torch.linalg.svd(fea, full_matrices=False)

#     U, S, Vh = torch.linalg.svd(fea, full_matrices=False)

#     S_mod = S.clone()
#     S_mod[:k_color] *= alpha

#     fea_texture = (U[:, :k_s] * S_mod[:k_s]) @ Vh[:k_s, :]

#     return fea_texture[:Num,:].unsqueeze(0) 



def svd_remove_color_sadis(feature_gray, k_color=6, alpha=0.1, rank_ratio=0.5):
    # =================================
    # 1. 输入特征
    F_gray = feature_gray.squeeze(0)      # [1374, 1024]
    F_gray = F_gray.float()               # 建议使用 float32

    Num, Dim = F_gray.shape[0], F_gray.shape[1]

    fea = F_gray
    k_s = int(Dim * rank_ratio)
    # 3. SVD
    # =================================
    # Uc, Sc, Vc = torch.linalg.svd(fea, full_matrices=False)

    U, S, Vh = torch.linalg.svd(fea, full_matrices=False)

    S_mod = S.clone()
    S_mod[:k_color] = 0.0

    fea_texture = (U[:, :k_s] * S_mod[:k_s]) @ Vh[:k_s, :]

    return fea_texture.unsqueeze(0) 


def remove_color_statistics(feature, eps=1e-5):
    # fea: [N, D]
    fea = feature.squeeze(0)      # [1374, 1024]
    fea = fea.float()               # 建议使用 float32
    mean = fea.mean(dim=0, keepdim=True)
    fea_c = fea - mean

    cov = fea_c.T @ fea_c / (fea_c.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)

    # whitening
    fea_w = fea_c @ eigvec @ torch.diag(1.0 / torch.sqrt(eigval + eps))

    return fea_w.unsqueeze(0) 


def wct_whiten(feature, eps=1e-5):
    fea = feature.squeeze(0)      # [1374, 1024]
    fea = fea.float()               # 建议使用 float32
    
    
    mean = fea.mean(dim=0, keepdim=True)
    fea_c = fea - mean

    cov = fea_c.T @ fea_c / (fea_c.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)

    fea_w = fea_c @ eigvec @ torch.diag(1.0 / torch.sqrt(eigval + eps))
    return fea_w.unsqueeze(0) 
    # return fea_w, eigvec, eigval, mean

def zca_whitening(feature, eps=1e-5):
    """
    fea: [N, D]  (N: 样本数, D: 特征维)
    return: ZCA-whitened features, same shape
    """
    fea = feature.squeeze(0)      # [1374, 1024]
    fea = fea.float()               # 建议使用 float32
    
    
    
    # 1. 去均值（颜色/风格的一阶统计）
    mean = fea.mean(dim=0, keepdim=True)
    fea_c = fea - mean

    # 2. 协方差矩阵
    cov = fea_c.T @ fea_c / (fea_c.shape[0] - 1)   # [D, D]

    # 3. 特征分解（协方差是对称矩阵）
    eigval, eigvec = torch.linalg.eigh(cov)

    # 4. Whitening
    diag = torch.diag(1.0 / torch.sqrt(eigval + eps))
    fea_white = fea_c @ eigvec @ diag @ eigvec.T

    return fea_white.unsqueeze(0) 


# def whiten_and_color(cF, sF):
#     # input: [1,c,h,w]  / [1,c,hw]
#     dtype = cF.dtype
#     device = cF.device
#     cF = cF.double()
#     sF = sF.double()
#     cF = cF.squeeze(0)
#     sF = sF.squeeze(0)
#     if len(cF.shape)==len(sF.shape) and len(sF.shape)==3:
#         C,W,H = cF.size(0),cF.size(1),cF.size(2)
#         _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
#         cF = cF.view(C,-1)
#         sF = sF.view(C,-1)

#     cFSize = cF.size()
#     c_mean = torch.mean(cF,1) # c x (h x w)
#     c_mean = c_mean.unsqueeze(1).expand_as(cF)
#     cF = cF - c_mean

#     contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double().to(device)
#     c_u,c_e,c_v = torch.svd(contentConv,some=False)

#     k_c = cFSize[0]
#     for i in range(cFSize[0]):
#         if c_e[i] < 0.00001:
#             k_c = i
#             break

#     sFSize = sF.size()
#     s_mean = torch.mean(sF,1)
#     sF = sF - s_mean.unsqueeze(1).expand_as(sF)
#     styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
#     s_u,s_e,s_v = torch.svd(styleConv,some=False)
#     k_s = sFSize[0]
#     for i in range(sFSize[0]):
#         if s_e[i] < 0.00001:
#             k_s = i
#             break
    
#     c_d = (c_e[0:k_c]).pow(-0.5)
#     step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
#     step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
#     whiten_cF = torch.mm(step2,cF)  

#     s_d = (s_e[0:k_s]).pow(0.5)

#     targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    
#     targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    
#     targetFeature = targetFeature.unsqueeze(0).to(dtype)


#     return targetFeature


# def whiten_and_color(sF, sf_gray):
#     # input: [1,c,h,w]  / [1,c,hw]
#     dtype = sF.dtype
#     device = sF.device
#     sF = sF.double()
#     sF = sF.squeeze(0) # [N, C]

#     sFSize = sF.size()
#     s_mean = torch.mean(sF,1)
#     sF = sF - s_mean.unsqueeze(1).expand_as(sF)
#     styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
#     s_u,s_e,s_v = torch.svd(styleConv,some=False)
#     k_s = sFSize[0]
#     for i in range(sFSize[0]):
#         if s_e[i] < 0.00001:
#             k_s = i
#             break
    
#     s_d = (s_e[0:k_s]).pow(0.5)

#     targetFeature = torch.mm(s_u[:,0:k_s],torch.diag(s_d)), (s_v[:,0:k_s].t())
    
    
#     targetFeature = targetFeature.unsqueeze(0).to(dtype)


#     return targetFeature