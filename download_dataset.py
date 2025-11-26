"""
自动下载并解压 BOUN Mouse Dynamics Dataset
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess

# 配置
DATASET_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/w6cxr8yc7p-2.zip"
DOWNLOAD_DIR = Path("downloads")
EXTRACT_DIR = Path(".")
DATASET_NAME = "boun-mouse-dynamics-dataset"

def download_file(url, output_path):
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        output_path: 保存路径
    """
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载进度") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✓ 下载完成: {output_path}")


def extract_zip(zip_path, extract_to):
    """
    解压 ZIP 文件并显示进度

    Args:
        zip_path: ZIP 文件路径
        extract_to: 解压目标目录
    """
    print(f"\n正在解压: {zip_path}")
    print(f"解压到: {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="解压进度") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

    print(f"✓ 解压完成: {zip_path}")


def extract_multipart_zip(base_path, extract_to):
    """
    解压分卷 ZIP 文件 (.zip, .z01, .z02, ...)

    Args:
        base_path: 主 ZIP 文件路径（.zip）
        extract_to: 解压目标目录
    """
    print(f"\n检测到分卷压缩文件: {base_path}")

    # 检查是否有 7z 工具（尝试多个命令）
    has_7z = False
    zip_cmd = None

    for cmd in ['7z', '7za', 'p7zip']:
        try:
            result = subprocess.run([cmd], capture_output=True, check=False)
            has_7z = True
            zip_cmd = cmd
            print(f"✓ 检测到 7-Zip 工具: {cmd}")
            break
        except FileNotFoundError:
            continue

    if not has_7z:
        print("⚠ 未检测到 7-Zip，尝试使用 Python 原生方法")

    extract_to.mkdir(parents=True, exist_ok=True)

    if has_7z:
        # 使用 7z 解压分卷文件
        print(f"使用 {zip_cmd} 解压...")
        cmd = [zip_cmd, 'x', str(base_path), f'-o{extract_to}', '-y']
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        for line in process.stdout:
            if line.strip():
                print(f"  {line.strip()}")

        process.wait()

        if process.returncode == 0:
            print(f"✓ 解压完成: {base_path}")
        else:
            print(f"✗ 解压失败: {process.stderr.read()}")
            raise Exception("7-Zip 解压失败")
    else:
        # 尝试使用 Python zipfile（可能不支持分卷）
        print("⚠ 警告: Python zipfile 可能不支持分卷压缩")
        print("建议安装 7-Zip: https://www.7-zip.org/download.html")

        try:
            with zipfile.ZipFile(base_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc="解压进度") as pbar:
                    for member in members:
                        zip_ref.extract(member, extract_to)
                        pbar.update(1)
            print(f"✓ 解压完成: {base_path}")
        except Exception as e:
            print(f"✗ Python zipfile 无法解压分卷文件: {e}")
            print("\n请手动安装 7-Zip 并重试:")
            print("  Windows: https://www.7-zip.org/download.html")
            print("  Linux: sudo apt-get install p7zip-full")
            print("  Mac: brew install p7zip")
            raise


def main():
    """主函数"""
    print("="*60)
    print("BOUN Mouse Dynamics Dataset 自动下载工具")
    print("="*60)

    # 检查数据集是否已存在
    if (EXTRACT_DIR / DATASET_NAME).exists():
        response = input(f"\n数据集已存在: {EXTRACT_DIR / DATASET_NAME}\n是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("取消下载")
            return
        print("删除现有数据集...")
        shutil.rmtree(EXTRACT_DIR / DATASET_NAME)

    # 步骤 1: 下载主 ZIP 文件
    main_zip = DOWNLOAD_DIR / "w6cxr8yc7p-2.zip"

    if main_zip.exists():
        print(f"\n主文件已存在: {main_zip}")
        response = input("是否跳过下载? (y/n): ")
        if response.lower() == 'y':
            print("跳过下载步骤")
        else:
            main_zip.unlink()
            download_file(DATASET_URL, main_zip)
    else:
        download_file(DATASET_URL, main_zip)

    # 步骤 2: 解压主 ZIP 文件
    temp_extract = DOWNLOAD_DIR / "temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    extract_zip(main_zip, temp_extract)

    # 步骤 3: 查找数据集的分卷文件
    print("\n查找数据集文件...")
    dataset_files = list(temp_extract.glob(f"{DATASET_NAME}.*"))

    if not dataset_files:
        print(f"✗ 未找到数据集文件: {DATASET_NAME}.*")
        print(f"可用文件: {list(temp_extract.iterdir())}")
        return

    print(f"找到 {len(dataset_files)} 个文件:")
    for f in sorted(dataset_files):
        print(f"  - {f.name}")

    # 步骤 4: 解压数据集（可能是分卷）
    main_dataset_zip = temp_extract / f"{DATASET_NAME}.zip"

    if main_dataset_zip.exists():
        # 检查是否有分卷文件 (.z01, .z02, ...)
        part_files = list(temp_extract.glob(f"{DATASET_NAME}.z*"))

        if part_files:
            print(f"\n检测到分卷压缩文件 ({len(part_files) + 1} 个分卷)")
            extract_multipart_zip(main_dataset_zip, EXTRACT_DIR)
        else:
            print("\n普通 ZIP 文件")
            extract_zip(main_dataset_zip, EXTRACT_DIR)
    else:
        print(f"✗ 未找到主数据集文件: {main_dataset_zip}")
        return

    # 步骤 5: 清理临时文件
    print("\n清理临时文件...")
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    # 可选：删除下载的文件
    response = input(f"\n是否删除下载的文件以节省空间? (y/n): ")
    if response.lower() == 'y':
        if main_zip.exists():
            main_zip.unlink()
            print(f"✓ 已删除: {main_zip}")
        if DOWNLOAD_DIR.exists() and not list(DOWNLOAD_DIR.iterdir()):
            DOWNLOAD_DIR.rmdir()
            print(f"✓ 已删除空目录: {DOWNLOAD_DIR}")

    # 验证最终结果
    final_dataset = EXTRACT_DIR / DATASET_NAME
    if final_dataset.exists():
        print("\n" + "="*60)
        print("✓ 数据集下载并解压成功!")
        print("="*60)
        print(f"数据集位置: {final_dataset.absolute()}")

        # 统计文件数量
        users_dir = final_dataset / "users"
        if users_dir.exists():
            user_folders = list(users_dir.iterdir())
            total_files = sum(1 for user_dir in user_folders
                            for test_type in ['internal_tests', 'external_tests']
                            for _ in (user_dir / test_type).glob('*.csv')
                            if (user_dir / test_type).exists())
            print(f"用户数量: {len(user_folders)}")
            print(f"CSV 文件总数: {total_files}")

        print("\n下一步:")
        print("  1. 运行数据清洗: python clean_data.py")
        print("  2. 运行诊断检查: python diagnose.py")
        print("  3. 开始训练: python train.py")
    else:
        print("\n✗ 数据集解压失败")
        print(f"预期位置: {final_dataset.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
