import trimesh
import pyvista as pv
import numpy as np
import angle_deg
#import open3d as o3d
import re
def load_and_preprocess_stl(path):
    """加载STL模型并预处理"""
    mesh = pv.read(path)
    mesh = mesh.clean()  # 清理重复点
    mesh = mesh.decimate(0.1) if mesh.n_faces > 10000 else mesh  # 简化大网格
    return mesh


def mirror_along_plane(mesh1, mesh2):

    x_center = mesh2.points[:, 0].mean()
    mirrored_mesh = mesh1.copy()
    points = mirrored_mesh.points
    points[:, 0] = 2 * x_center - points[:, 0]
    mirrored_mesh.points = points
    return mirrored_mesh


# def pyvista_to_trimesh(pyvista_mesh):
#     """将pyvista网格转换为trimesh网格"""
#     faces = pyvista_mesh.faces
#     vertices = pyvista_mesh.points
#
#     # 确保faces是一维数组
#     if faces.ndim != 1:
#         raise ValueError("Expected faces to be a 1D array.")
#
#     # 初始化trimesh的面列表
#     trimesh_faces = []
#
#     # 假设每个面是三角形（3个顶点）
#     num_vertices_per_face = 3
#     num_faces = faces.size // num_vertices_per_face
#
#     for i in range(num_faces):
#         # 提取三角形的顶点索引
#         start_index = i * num_vertices_per_face
#         end_index = start_index + num_vertices_per_face
#         face = faces[start_index:end_index].tolist()
#         trimesh_faces.append(face)
#
#     # 创建trimesh对象
#     return trimesh.Trimesh(vertices=vertices, faces=trimesh_faces)

def pyvista_to_trimesh(pyvista_mesh):
    """将pyvista网格转换为trimesh网格，保持水密性"""
    # 获取顶点和面数据
    vertices = pyvista_mesh.points
    faces = pyvista_mesh.faces

    # 检查面数据格式
    if faces.ndim != 1:
        raise ValueError("Expected faces to be a 1D array.")

    # PyVista的面数据格式是 [n_vertices, v1, v2, ..., vn, n_vertices, ...]
    # 需要解析为 [[v1, v2, v3], [v4, v5, v6], ...]
    trimesh_faces = []
    i = 0
    while i < len(faces):
        n_vertices = faces[i]  # 当前面的顶点数
        if n_vertices != 3:
            raise ValueError("Only triangular meshes are supported.")
        face_vertices = faces[i+1 : i+1+n_vertices].tolist()
        trimesh_faces.append(face_vertices)
        i += 1 + n_vertices  # 移动到下一个面

    # 创建trimesh对象
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=trimesh_faces)

    # 验证水密性
    if not trimesh_mesh.is_watertight:
        print("警告：转换后的网格可能不水密，尝试修复...")
        try:
            trimesh_mesh = trimesh.repair.fill_holes(trimesh_mesh, max_hole_size=1e5)
        except Exception as e:
            print(f"修复失败: {str(e)}")

    return trimesh_mesh

def ensure_watertight_legacy(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("输入必须为trimesh对象")

    # 创建副本以避免修改原始网格
    repaired_mesh = mesh.copy()

    # 第一步：基础修复 (兼容旧版本)
    print("执行基础修复...")
    trimesh.repair.fix_normals(repaired_mesh)

    # 移除退化面（旧版本参数不同）
    #trimesh.repair.remove_degenerate_faces(repaired_mesh, check_tex=False)
    broken=trimesh.repair.broken_faces(repaired_mesh)
    if broken.any():
        repaired_mesh=trimesh.repair.fill_holes(repaired_mesh)

    # 合并重复顶点（旧版本需要显式调用）
    #trimesh.repair.merge_vertices(repaired_mesh)
    vertices,faces=trimesh.remesh.subdivide(repaired_mesh.vertices,repaired_mesh.faces)
    repaired_mesh=trimesh.Trimesh(vertices=vertices,faces=faces)

    # 第二步：填充孔洞（旧版本特殊处理）
    def ensure_watertight_legacy(mesh):
        repaired_mesh = mesh.copy()

        # 第一步：检查并填充孔洞
        if not repaired_mesh.is_watertight:
            print("尝试填充孔洞...")
            try:
                # 3.6.8需要显式指定inplace=False
                repaired_mesh = repaired_mesh.fill_holes(
                    max_hole_size=1e5,  # 设置大尺寸阈值
                    inplace=False  # 关键参数！
                )
            except Exception as e:
                print(f"孔洞填充失败: {str(e)}")

        # 第二步：处理非流形几何
        if not repaired_mesh.is_watertight:
            print("处理非流形边...")
            try:
                # 使用trimesh.repair中的函数修复非流形几何
                repaired_mesh = trimesh.repair.fix_normals(repaired_mesh)
                repaired_mesh = trimesh.repair.split_concave(repaired_mesh)
                repaired_mesh = trimesh.repair.merge_vertices(repaired_mesh)
            except Exception as e:
                print(f"非流形边处理失败: {str(e)}")

        return repaired_mesh

        # 分离最大连体组件（旧版本接口）
        components = trimesh.graph.connected_components(
            repaired_mesh.edges,
            nodes=np.arange(len(repaired_mesh.vertices)),
            min_len=3
        )
        if len(components) > 0:
            # 提取最大组件
            largest = max(components, key=len)
            repaired_mesh = repaired_mesh.submesh([largest])[0]

    # 第四步：二次验证（旧版本检测方式）
    is_sealed = repaired_mesh.is_watertight
    if is_sealed:
        print("修复成功：网格已闭合")
    else:
        print("警告：未能完全修复水密性")
        # 显示问题区域
        if hasattr(repaired_mesh, 'show'):
            repaired_mesh.show()

    return repaired_mesh

def rotate_around_custom_axis(mesh, best_theta_deg, principal_axis):
    """绕指定轴（principal_axis）旋转网格，旋转角度为best_theta_deg（角度制）"""
    if mesh is None:
        raise ValueError("输入的网格对象是None，请检查前面的操作。")

    if len(principal_axis) != 3:
        raise ValueError("旋转轴principal_axis必须是三维向量。")

    # 将角度从度数转换为弧度
    best_theta_rad = np.radians(best_theta_deg)

    # 归一化旋转轴
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    # 罗德里格旋转公式
    ux, uy, uz = principal_axis
    sin_theta = np.sin(best_theta_rad)
    cos_theta = np.cos(best_theta_rad)
    one_minus_cos_theta = 1 - cos_theta

    rotation_matrix_3x3 = np.array([
        [cos_theta + ux ** 2 * one_minus_cos_theta, ux * uy * one_minus_cos_theta - uz * sin_theta,
         ux * uz * one_minus_cos_theta + uy * sin_theta],
        [uy * ux * one_minus_cos_theta + uz * sin_theta, cos_theta + uy ** 2 * one_minus_cos_theta,
         uy * uz * one_minus_cos_theta - ux * sin_theta],
        [uz * ux * one_minus_cos_theta - uy * sin_theta, uz * uy * one_minus_cos_theta + ux * sin_theta,
         cos_theta + uz ** 2 * one_minus_cos_theta]
    ])

    # 将3x3旋转矩阵扩展为4x4矩阵
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3

    # 进行旋转
    transformed_mesh = mesh.transform(rotation_matrix_4x4)

    return transformed_mesh


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 加载数据
    impacted_tooth = load_and_preprocess_stl(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-05\11 contralateral tooth.stl")
    mesh_contralateral=load_and_preprocess_stl(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-05\21 impacted tooth.stl")
    alveolar_bone = load_and_preprocess_stl(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-05\alveolar bone.stl")

    # 2. 镜像对称处理
    mirrored_tooth = mirror_along_plane(mesh_contralateral, alveolar_bone)

    mirrored_tooth_trimesh = pyvista_to_trimesh(mirrored_tooth)
    mesh_contralateral_trimesh = pyvista_to_trimesh(mesh_contralateral)
    mirrored_tooth_trimesh.show()
    mesh_contralateral_trimesh.show()
    mesh_contralateral_trimesh=ensure_watertight_legacy(mesh_contralateral_trimesh)
    mirrored_tooth_trimesh=ensure_watertight_legacy(mirrored_tooth_trimesh)
    #搜索最佳旋转角度


    result =angle_deg.align_and_rotate(mirrored_tooth_trimesh,mesh_contralateral_trimesh)
    principal_axis=result['rotation_axis']
    best_theta=result['rotation_angle']
    print(f"Rotation Axis: {principal_axis}")
    print(type(principal_axis))
    print(f"Rotation Angle: {result['rotation_angle']} degrees")
    # 3. 绕Y轴旋转30度（假设中线为Y轴方向）
    if isinstance(principal_axis, str):
        # 去掉方括号并分割字符串
        parts = principal_axis.strip("[]").split(",")
        # 转换为浮点数列表
        principal_axis = [float(x.strip()) for x in parts]
    print(type(principal_axis))
    print(type(best_theta))
    rotated_tooth = rotate_around_custom_axis(impacted_tooth, best_theta, principal_axis)

    # 4. 布尔差集运算（需确保网格闭合）
    try:
        difference = rotated_tooth.boolean_difference(alveolar_bone)
    except RuntimeError:
        # 如果布尔运算失败，尝试修复网格
        rotated_tooth = rotated_tooth.repair()
        alveolar_bone = alveolar_bone.repair()
        difference = rotated_tooth.boolean_difference(alveolar_bone)

    # 5. 可视化
    plotter = pv.Plotter(shape=(2, 3), window_size=(1920, 1080))

    # 原始阻生牙
    plotter.subplot(0, 0)
    plotter.add_mesh(impacted_tooth, color="gold", show_edges=True)
    plotter.add_title(f"Original Impacted Tooth{impacted_tooth.volume:.2f}")

    # 镜像后的阻生牙
    plotter.subplot(0, 1)
    plotter.add_mesh(mirrored_tooth, color="cyan", show_edges=True)
    plotter.add_title("Mirrored Tooth")

    # 旋转后的阻生牙
    plotter.subplot(0, 2)
    plotter.add_mesh(rotated_tooth, color="magenta", show_edges=True)
    plotter.add_title(f"{principal_axis} {best_theta} ")

    # 牙槽骨
    plotter.subplot(1, 0)
    plotter.add_mesh(alveolar_bone, color="bisque", opacity=0.8)
    plotter.add_title("Alveolar Bone")

    # 差集结果
    plotter.subplot(1, 1)
    plotter.add_mesh(difference, color="red", opacity=0.9)
    plotter.add_title(f"Difference Volume: {difference.volume:.2f} mm³")

    # 组合视图
    plotter.subplot(1, 2)
    plotter.add_mesh(alveolar_bone, color="bisque", opacity=0.3)
    plotter.add_mesh(rotated_tooth, color="magenta", opacity=0.5)
    plotter.add_mesh(difference, color="red")
    plotter.add_title("Combined View")

   # plotter.link_views()  # 同步所有视图的相机角度
    plotter.show();