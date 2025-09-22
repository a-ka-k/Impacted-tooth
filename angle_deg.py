import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

def align_and_rotate(mesh_impacted, mesh_contralateral, coarse_step=5, fine_step=1):
    """
    对齐阻生牙和对侧同名牙的惯性线，并旋转阻生牙以获得最大重合体积。

    参数:
        mesh_impacted (trimesh.Trimesh): 阻生牙的网格
        mesh_contralateral (trimesh.Trimesh): 对侧同名牙的网格
        coarse_step (float): 粗搜索的旋转步长（度）
        fine_step (float): 细搜索的旋转步长（度）

    返回:
        dict: 包含旋转轴、旋转角度和变换后的阻生牙网格的字典
    """

    def get_centroid_principal_axes(mesh):
        """计算网格的质心和主惯性轴"""
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Input must be a trimesh.Trimesh object.")
        vertices = mesh.vertices
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        return centroid, eigenvectors[:, order]

    # 获取质心和主惯性轴
    centroid1, axes1 = get_centroid_principal_axes(mesh_impacted)
    centroid2, axes2 = get_centroid_principal_axes(mesh_contralateral)

    # 调整阻生牙的主轴方向以匹配对侧牙
    for i in range(3):
        if np.dot(axes1[:, i], axes2[:, i]) < 0:
            axes1[:, i] *= -1

    # 计算旋转矩阵并变换阻生牙顶点
    R_matrix = axes2 @ axes1.T
    translated_vertices = mesh_impacted.vertices - centroid1
    rotated_vertices = (translated_vertices @ R_matrix.T) + centroid2

    # 确定旋转轴（对侧牙的第一主轴）
    principal_axis = axes2[:, 0].copy()
    principal_axis /= np.linalg.norm(principal_axis)

    def rotate_vertices(vertices, theta):
        """绕指定轴旋转顶点"""
        angle_rad = np.radians(theta)
        rotation = R.from_rotvec(principal_axis * angle_rad)
        translated = vertices - centroid2
        return rotation.apply(translated) + centroid2

    def evaluate_volume(theta):
        """评估旋转角度下的交集体积"""
        rotated_v = rotate_vertices(rotated_vertices, theta)
        temp_mesh = trimesh.Trimesh(vertices=rotated_v, faces=mesh_impacted.faces)
        intersection = temp_mesh.intersection(mesh_contralateral)
        return intersection.volume if not intersection.is_empty else 0

    # 粗搜索最佳角度
    max_vol, best_theta = -np.inf, 0
    for theta in np.arange(0, 360, coarse_step):
        vol = evaluate_volume(theta)
        if vol > max_vol:
            max_vol, best_theta = vol, theta

    # 细搜索优化
    for theta in np.linspace(best_theta - coarse_step, best_theta + coarse_step, 100):
        vol = evaluate_volume(theta)
        if vol > max_vol:
            max_vol, best_theta = vol, theta

    # 生成最终网格
    best_vertices = rotate_vertices(rotated_vertices, best_theta)
    result_mesh = trimesh.Trimesh(vertices=best_vertices, faces=mesh_impacted.faces)

    return {
        'rotation_axis': principal_axis,
        'rotation_angle': best_theta,
    }


if __name__ == "__main__":
    mesh_impacted = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\13镜像.stl")
    mesh_contralateral = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\fujinjing 13 contralateral tooth.stl")

    result = align_and_rotate(mesh_impacted, mesh_contralateral)
    print(f"Rotation Axis: {result['rotation_axis']}")
    print(f"Rotation Angle: {result['rotation_angle']} degrees")
    result['transformed_mesh'].export('aligned_impacted_tooth.stl')