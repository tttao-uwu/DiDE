import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple, Union
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_pca_3d(feature_tensor, n_components=3, thresh=0.4, path=None):
    """
    feature_tensor: numpy array, shape (1, 8, 16, 16, 16)
    n_components: PCA 
    thresh: Visualize intensity (show only locations with high energy)
    path: 
    """
    feature = feature_tensor.squeeze(0)  # (8, 16, 16, 16)
    C, D, H, W = feature.shape
    data = feature.reshape(C, -1).T   # (4096, 8)
    # PCA
    data = data.cpu()
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)   # (4096, n_components)
    data_pca = data_pca.reshape(D, H, W, n_components)


    # n_components == 3
    value = data_pca
    value_min = value.min((0,1,2), keepdims=True)
    value_max = value.max((0,1,2), keepdims=True)
    value_norm = (value - value_min) / (value_max - value_min + 1e-8)
    intens = np.linalg.norm(value_norm, axis=-1)  #  Visualize intensity
    mask = intens > thresh
    xs, ys, zs = np.where(mask)
    # cs = value_norm[mask]
    # cs = np.clip(cs, 0, 1)
    cs = value_norm[mask]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=cs, s=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.colorbar(sc, label='PCA Value')

    if path is not None:
        plt.savefig(path)
    plt.close(fig)


def visualize_latent_decoder_3d(feature_tensor, thresh=0.3, cmap: str = "viridis", path=None):
    """
    feature_tensor: numpy array, shape (1, 1, 64, 64, 64)
    n_components: PCA 
    thresh: Visualize intensity (show only locations with high energy)
    path: 
    """
    feature = feature_tensor.squeeze(0).squeeze(0)  # (64, 64, 64)
    data = feature.cpu().numpy()
    # n_components == 3
    value = data
    # normalize min max
    vmin = value.min()
    vmax = value.max()
    value_norm = (value - vmin) / (vmax - vmin + 1e-8)
    
    # # # normalize softmax
    # Tm = 0.5  # <1 更尖锐

    # v = value.reshape(-1)
    # v = v / Tm

    # v_soft = np.exp(v)
    # v_soft = v_soft / (v_soft.sum() + 1e-8)

    # value_norm = v_soft.reshape(value.shape)
    

    # Apply threshold
    mask = value_norm > thresh


    xs, ys, zs = np.where(mask)
    colors = value_norm[mask]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=colors, cmap=cmap, s=2, alpha=0.5)

    fig.colorbar(sc, ax=ax, shrink=0.6, label='Normalized Attn')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Latent Decoder Feature')

    plt.tight_layout()
    plt.savefig(os.path.join(path))
    plt.close(fig)

 

def visualize_features(features_list, n_components=3, thresh=0.3, save_dir='feature_images'):
    os.makedirs(save_dir, exist_ok=True)
    for i, feat in enumerate(features_list):
        print(f'Saving sample {i+1}/{len(features_list)}')
        save_path = os.path.join(save_dir, f'feature_{i:02d}.png')
        visualize_pca_3d(
            feat if isinstance(feat, np.ndarray) else feat.cpu().numpy(),
            n_components=n_components,
            thresh=thresh,
            save_path=save_path
        )


def visualize_features_slat(features_list, path='feature_images'):
    
    for i, feat in enumerate(features_list):

        # scaler = StandardScaler()
        # feats_scaled = scaler.fit_transform(feat.feats.cpu().numpy())
        # pca = PCA(n_components=3)
        # feats_pca = pca.fit_transform(feats_scaled)

        pca = PCA(n_components=3)
        feats_pca = pca.fit_transform(feat.feats.cpu().numpy())

        value_min = feats_pca.min((0), keepdims=True)
        value_max = feats_pca.max((0), keepdims=True)
        feats_pca = (feats_pca - value_min) / (value_max - value_min + 1e-8)
        # feats_pca = np.linalg.norm(feats_pca, axis=-1)  # 

        coords = feat.coords.cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        color = feats_pca[:, :]
        sc = ax.scatter(coords[:, 1], coords[:, 2], coords[:, 3], c=color, s=8)
        plt.title('3D Spatial Distribution Colored by PCA Feature (PC1)')
        plt.tight_layout()
        plt.colorbar(sc, label='PC1 Value')
        plt.show()
        plt.savefig(path)
        plt.close(fig)


def visualize_cross_attn_bin(
                         cross_attn, 
                         thresh=0.3,
                         save_dir=None,
                         ):
    
    cross_attn = cross_attn.reshape(16, 16, 16, -1)

    for i in range(1,2):

        cross_attn_ins = cross_attn[:, :, :, i].cpu()
        value = np.array(cross_attn_ins, dtype=float)
        value_min = value.min((0,1,2), keepdims=True)
        value_max = value.max((0,1,2), keepdims=True)
        value_norm = (value - value_min) / (value_max - value_min + 1e-8)
        
        x, y, z = np.indices(value_norm.shape)
        
        
        # flatten
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = value_norm.flatten()
        
        # mask
        mask_high = values > 0.5     # 大于 0.5
        mask_low  = values <= 0.5   # 小于等于 0.5

        # create 3d fig
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # black for low value
        ax.scatter(x[mask_low], y[mask_low], z[mask_low],
                c='black', s=5, alpha=0.3)

        # color for high value
        ax.scatter(x[mask_high], y[mask_high], z[mask_high],
                        c='red', cmap='viridis', s=10, alpha=0.9)

    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'feature_{i}.png')
        plt.savefig(save_path)
        plt.close(fig)


def visualize_cross_attn(
    cross_attn,
    threshold=0.0,
    cmap: str = "viridis",
    save_dir=None,
):
    # os.makedirs(save_dir, exist_ok=True)

    cross_attn = cross_attn.reshape(16, 16, 16, -1)

    for i in range(2, 3):

        value = cross_attn[:, :, :, i].detach().cpu().numpy()

        # normalize min max
        vmin = value.min()
        vmax = value.max()
        value_norm = (value - vmin) / (vmax - vmin + 1e-8)
        
        # # # normalize softmax
        # Tm = 0.5  # <1 更尖锐

        # v = value.reshape(-1)
        # v = v / Tm

        # v_soft = np.exp(v)
        # v_soft = v_soft / (v_soft.sum() + 1e-8)

        # value_norm = v_soft.reshape(value.shape)
        
    
        # Apply threshold
        mask = value_norm > threshold


        xs, ys, zs = np.where(mask)
        colors = value_norm[mask]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(xs, ys, zs, c=colors, cmap=cmap, s=10, alpha=0.9)

        fig.colorbar(sc, ax=ax, shrink=0.6, label='Normalized Attn')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cross-Attention Feature {i}')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir))
        plt.close(fig)


def visualize_cross_attn_value_5(
    cross_attn,
    threshold=0.0,
    cmap: str = "viridis",
    save_dir=None,
):
    # os.makedirs(save_dir, exist_ok=True)

    cross_attn = cross_attn.reshape(16, 16, 16, -1)

    for i in range(2, 3):

        value = cross_attn[:, :, :, i].detach().cpu().numpy()

        # normalize min max
        vmin = value.min()
        vmax = value.max()
        value_norm = (value - vmin) / (vmax - vmin + 1e-8)
        

        # ----------------------------
        # 2. 按数值大小分成 5 类（等分位）
        # ----------------------------
        flat = value_norm.flatten()

        # 计算 5 类的分位数边界
        bins = np.quantile(flat, [0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # digitize：得到类别编号 0~4
        labels = np.digitize(flat, bins[1:-1], right=True)

        # 还原回 3D
        labels_3d = labels.reshape(value_norm.shape)

        # ----------------------------
        # 3. 生成 3D 坐标
        # ----------------------------
        x, y, z = np.meshgrid(
            np.arange(16),
            np.arange(16),
            np.arange(16),
            indexing="ij"
        )

        xs = x.flatten()
        ys = y.flatten()
        zs = z.flatten()
        colors = labels_3d.flatten()

        # ----------------------------
        # 4. 定义 5 种离散颜色
        # ----------------------------
        cmap = ListedColormap([
            "#0B0B0B",  # 深紫
            "#f0d805",  # 蓝
            "#0E19E8",  # 青
            "#5ec962",  # 绿
            "#F7060E"   # 黄
        ])

        norm = BoundaryNorm(boundaries=np.arange(-0.5, 5.5, 1), ncolors=5)

        # ----------------------------
        # 5. 画 3D scatter
        # ----------------------------
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            xs, ys, zs,
            c=colors,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.9
        )

        # colorbar（显示类别）
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, ticks=range(5))
        cbar.set_label("Value Class (Low → High)")
        cbar.set_ticklabels([f"Class {i}" for i in range(5)])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("16*16*16 Tensor Classified into 5 Classes")

        plt.tight_layout()


        plt.savefig(os.path.join(save_dir))
        plt.close(fig)


def visualize_self_attn_bin(
                         self_attn, 
                         n_components = 3, 
                         save_dir=None,
                         ):
    
    self_attn_ins = self_attn.reshape(16, 16, 16, -1)
    D, H, W, C = self_attn_ins.shape
    # PCA
    data = self_attn.cpu()
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)   # (4096, n_components)
    data_pca = data_pca.reshape(D, H, W, n_components)

    # n_components == 3
    value = data_pca
    value_min = value.min(axis=(0,1,2), keepdims=True)
    value_max = value.max(axis=(0,1,2), keepdims=True)
    value_norm = (value - value_min) / (value_max - value_min + 1e-8)

    x, y, z, n = np.indices(value_norm.shape)
    
    # flatten
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = value_norm.flatten()
    
    # mask
    mask_high = values > 0.5     # 大于 0.5
    mask_low  = values <= 0.5   # 小于等于 0.5

    # create 3d fig
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # black for low value
    ax.scatter(x[mask_low], y[mask_low], z[mask_low],
            c='black', s=5, alpha=0.3)

    # color for high value
    ax.scatter(x[mask_high], y[mask_high], z[mask_high],
                    c='red', cmap='viridis', s=10, alpha=0.9)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    # plt.colorbar(sc, label='PCA Value')

    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close(fig)
    
def visualize_self_attn(
                         self_attn, 
                         n_components = 3, 
                         thresh=0.0,
                         save_dir=None,
                         ):
    
    self_attn_ins = self_attn.reshape(16, 16, 16, -1)
    D, H, W, C = self_attn_ins.shape
    # PCA
    data = self_attn.cpu()
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)   # (4096, n_components)
    data_pca = data_pca.reshape(D, H, W, n_components)

    # n_components == 3
    value = data_pca
    value_min = value.min(axis=(0,1,2), keepdims=True)
    value_max = value.max(axis=(0,1,2), keepdims=True)
    value_norm = (value - value_min) / (value_max - value_min + 1e-8)
    intens = np.linalg.norm(value_norm, axis=-1)  #  Visualize intensity
    mask = intens > thresh
    xs, ys, zs = np.where(mask)
    # cs = value_norm[mask]
    # cs = np.clip(cs, 0, 1)
    cs = value_norm[mask]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=cs, s=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.colorbar(sc, label='PCA Value')


    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close(fig)    

    
# def visualize_self_attn_3_face(
#                                 self_attn, 
#                                 thresh=0.3,
#                                 n_components = 3, 
#                                 save_dir=None,
#                                 ):
    
#     self_attn = self_attn.reshape(16, 16, 16, -1)
#     D, H, W, C = self_attn.shape
#     # PCA
#     data = self_attn.cpu()
#     pca = PCA(n_components=n_components)
#     data_pca = pca.fit_transform(data)   # (4096, n_components)
#     data_pca = data_pca.reshape(D, H, W, n_components)

#     # n_components == 3
#     value = data_pca
#     value_min = value.min((0,1,2), keepdims=True)
#     value_max = value.max((0,1,2), keepdims=True)
#     value_norm = (value - value_min) / (value_max - value_min + 1e-8)
#     intens = np.linalg.norm(value_norm, axis=-1)  #  Visualize intensity
#     mask = intens > thresh
#     xs, ys, zs = np.where(mask)

#     cs = value_norm[mask]

#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     sc = ax.scatter(xs, ys, zs, c=cs, s=8)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.tight_layout()
#     plt.colorbar(sc, label='PCA Value')

#     if save_dir is not None:
#         plt.savefig(save_dir)
#     plt.close(fig)



def visualize_cross_attn_slat(
                         cross_attn, 
                         coords_h,
                         save_dir=None,
                         ):

    for i in range(1,3):
        cross_attn_ins = cross_attn[:,i].unsqueeze(-1).cpu()
        value = np.array(cross_attn_ins, dtype=float)

        value_min = value.min((0), keepdims=True)
        value_max = value.max((0), keepdims=True)
        value_norm = (value - value_min) / (value_max - value_min + 1e-8)

        coords = np.array(coords_h.cpu())

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        color = value_norm[:, :]
        sc = ax.scatter(coords[:, 1], coords[:, 2], coords[:, 3], c=color, s=8)
        save_path = os.path.join(save_dir, f'feature_{i}.png')
        
        plt.title('3D Spatial Distribution Colored by PCA Feature (PC1)')
        plt.tight_layout()
        plt.colorbar(sc, label='PC1 Value')
        plt.show()
        plt.savefig(save_path)
        plt.close(fig)




def visualize_self_attn_slat(
                         self_attn, 
                         coords_h, 
                         n_components = 3, 
                         save_dir=None,
                         ):
    

    # PCA
    data = self_attn.detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)   # (4096, n_components)
    
    # n_components == 3
    value = data_pca
    value_min = value.min(axis=(0), keepdims=True)
    value_max = value.max(axis=(0), keepdims=True)
    value_norm = (value - value_min) / (value_max - value_min + 1e-8)

    coords = np.array(coords_h.cpu())

    color = value_norm[:, :]
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 1], coords[:, 2], coords[:, 3], c=color, s=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.colorbar(sc, label='PCA Value')

    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close(fig)