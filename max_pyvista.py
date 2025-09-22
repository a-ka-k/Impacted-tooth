import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def visualize_alignment(mesh_impacted, mesh_contralateral):
    """
    可视化对齐和旋转过程的完整流程
    包含：初始位置、主轴显示、对齐后位置、旋转优化过程、最终结果
    """
    # 创建PyVista绘图窗口
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))

    # 转换trimesh到pyvista格式
    def trimesh_to_pv(mesh, color):
        return pv.wrap_trimesh(mesh).extract_surface().clean().triangulate().cast_to_unstructured_grid()

    # 原始位置可视化 (左上)
    plotter.subplot(0, 0)
    pv_initial_impacted = trimesh_to_pv(mesh_impacted, 'gold')
    pv_contralateral = trimesh_to_pv(mesh_contralateral, 'lightblue')

    plotter.add_mesh(pv_initial_impacted, color='gold', opacity=0.8, label='Impacted')
    plotter.add_mesh(pv_contralateral, color='lightblue', opacity=0.8, label='Contralateral')
    plotter.add_title("Initial Position", font_size=16)
    plotter.add_legend()

    # 计算惯性轴和变换矩阵
    def get_principal_axes(mesh):
        vertices = mesh.vertices - mesh.vertices.mean(0)
        cov = np.cov(vertices.T)
        return np.linalg.eigh(cov)[1][:, ::-1]

    axes_impacted = get_principal_axes(mesh_impacted)
    axes_contralateral = get_principal_axes(mesh_contralateral)

    # 对齐主轴后的位置 (右上)
    plotter.subplot(0, 1)
    aligned_mesh = mesh_impacted.copy()
    aligned_mesh.vertices = (mesh_impacted.vertices - mesh_impacted.vertices.mean(
        0)) @ axes_contralateral.T + mesh_contralateral.vertices.mean(0)
    pv_aligned = trimesh_to_pv(aligned_mesh, 'gold')

    # 可视化主轴
    centroid = mesh_contralateral.vertices.mean(0)
    for i, (color, length) in enumerate(zip(['red', 'green', 'blue'], [3, 2, 1])):
        axis = axes_contralateral[:, i] * length
        plotter.add_arrows(centroid, axis.reshape(1, 3), color=color, label=f'Principal Axis {i + 1}')

    plotter.add_mesh(pv_aligned, color='gold', opacity=0.8)
    plotter.add_mesh(pv_contralateral, color='lightblue', opacity=0.8)
    plotter.add_title("Aligned Principal Axes", font_size=16)
    plotter.add_legend()

    # 旋转优化过程可视化 (左下)
    plotter.subplot(1, 0)

    # 预计算旋转范围
    angles = np.linspace(0, 360, 36)
    volumes = []

    # 创建动画回调
    pv_rotating = pv_aligned.copy()
    actor = plotter.add_mesh(pv_rotating, color='gold', opacity=0.8)
    plotter.add_mesh(pv_contralateral, color='lightblue', opacity=0.8)
    plotter.add_title("Rotation Optimization", font_size=16)
    plotter.add_text("Press 'r' to start rotation animation", position='lower_edge')

    def animate():
        max_vol = 0
        best_angle = 0
        for theta in angles:
            # 执行旋转
            rot = R.from_rotvec(axes_contralateral[:, 0] * np.radians(theta))
            rotated_verts = rot.apply(aligned_mesh.vertices - centroid) + centroid
            pv_rotating.points = rotated_verts

            # 计算交集体积
            temp_mesh = trimesh.Trimesh(rotated_verts, aligned_mesh.faces)
            intersection = temp_mesh.intersection(mesh_contralateral)
            vol = intersection.volume if not intersection.is_empty else 0
            volumes.append(vol)

            # 更新最大体积
            if vol > max_vol:
                max_vol = vol
                best_angle = theta

            # 更新图形
            plotter.update_coordinates(pv_rotating.points, actor)
            plotter.update()

        return best_angle

    plotter.add_key_event('r', animate)

    # 最终结果可视化 (右下)
    plotter.subplot(1, 1)
    best_angle = animate()  # 执行完整优化
    pv_final = trimesh_to_pv(pv_rotating, 'gold')

    # 可视化交集体积
    intersection_mesh = trimesh.boolean.intersection([aligned_mesh, mesh_contralateral])
    if not intersection_mesh.is_empty:
        pv_intersection = trimesh_to_pv(intersection_mesh, 'red')
        plotter.add_mesh(pv_intersection, color='red', label='Intersection')

    plotter.add_mesh(pv_final, color='gold', opacity=0.8)
    plotter.add_mesh(pv_contralateral, color='lightblue', opacity=0.8)
    plotter.add_title(f"Optimal Position: {best_angle:.1f}°\nIntersection Volume: {max(volumes):.2f} mm³",
                      font_size=16)
    plotter.add_legend()

    # 显示所有视图
    plotter.show()


# 使用示例
if __name__ == "__main__":
    # 加载网格并预处理
    mesh_impacted = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\fujinjing 23 impacted_tooth.stl")
    mesh_contralateral = trimesh.load(r"C:\Users\Lenovo\Documents\WeChat Files\wxid_89te4g5q7xtn22\FileStorage\File\2025-03\fujinjing 13 contralateral tooth.stl")

    # 确保网格闭合
    for mesh in [mesh_impacted, mesh_contralateral]:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)

    visualize_alignment(mesh_impacted, mesh_contralateral)