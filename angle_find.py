import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2


def align_and_rotate(mesh_impacted, mesh_contralateral,
                     coarse_step=5, fine_step=1,
                     output_video='rotation.mp4',
                     resolution=(1920, 1080), fps=30):
    """
    直接生成旋转过程可视化视频（修正版）

    参数:
        output_video (str): 输出视频路径，支持.mp4/.avi
        resolution (tuple): 视频分辨率 (宽, 高)
        fps (int): 视频帧率
    """
    # 初始化视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'avc1'等编码
    video_writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        resolution
    )

    # 创建离屏渲染器
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=resolution[0],
        height=resolution[1],
        visible=False  # 无头模式运行
    )

    # 转换网格到Open3D格式
    def create_o3d_mesh(trimesh_obj, color):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh

    # 添加对侧牙（青蓝色）
    contralateral_o3d = create_o3d_mesh(mesh_contralateral, [0.2, 0.8, 0.8])
    vis.add_geometry(contralateral_o3d)

    # 添加阻生牙（珊瑚色）
    impacted_o3d = create_o3d_mesh(mesh_impacted, [0.9, 0.4, 0.3])
    vis.add_geometry(impacted_o3d)

    # 设置固定视角（基于主惯性轴）
    view_ctl = vis.get_view_control()
    camera_params = o3d.io.read_pinhole_camera_parameters("initial_camera.json")
    if camera_params:
        view_ctl.convert_from_pinhole_camera_parameters(camera_params)

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



    # 旋转优化帧捕获
    def capture_frame():
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        return (np.asarray(image) * 255).astype(np.uint8)[:, :, ::-1]  # RGB转BGR

    # 执行优化并录制
    for angle in optimization_angles:
        # 更新阻生牙位置
        new_vertices = compute_rotation(angle)
        impacted_o3d.vertices = o3d.utility.Vector3dVector(new_vertices)
        vis.update_geometry(impacted_o3d)

        # 捕获并写入帧
        frame = capture_frame()
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    vis.destroy_window()

    return {
        'rotation_axis': principal_axis,
        'rotation_angle': best_theta,
    }

# 加载STL模型
impacted = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\23.stl")
contralateral = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\fujinjing 13 contralateral tooth.stl")

# 生成手术规划视频
align_and_rotate(
    impacted,
    contralateral,
    output_video='\D:\PythonProject\crash\angle_find video\surgery_simulation.mp4',
    resolution=(2560, 1440),  # 2K分辨率
    fps=60                    # 流畅帧率
)

