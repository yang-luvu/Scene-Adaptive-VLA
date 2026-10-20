import os
import json
import glob

# Source evaluation directory.
viz_root = ""
# Normalized directory consumed by the metrics script.
target_metric_dir = ""

os.makedirs(target_metric_dir, exist_ok=True)

metric_files = glob.glob(os.path.join(viz_root, "*/debug_viz/*/*/*/metric/metric_info.json"))

print(f"找到 {len(metric_files)} 个物理指标文件，开始建立映射...")

for src_path in metric_files:
    # Strip the three-digit route ID from the source directory name.
    parts = src_path.split('/')
    viz_folder_name = ""
    for p in parts:
        if "RouteScenario" in p:
            viz_folder_name = p
            break
    
    if not viz_folder_name:
        continue
        
    save_name = viz_folder_name[3:] 
    
    save_dir = os.path.join(target_metric_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    dst_path = os.path.join(save_dir, "metric_info.json")
    if not os.path.exists(dst_path):
        os.symlink(src_path, dst_path)

print(f"映射完成！现在你可以运行统计脚本了。")
