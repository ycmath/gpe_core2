#!/usr/bin/env python
"""Setup script for GPE Core library.

대안 설치 스크립트 - pyproject.toml이 작동하지 않을 경우 사용
"""
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# 프로젝트 루트 디렉토리
here = Path(__file__).parent.resolve()

# README 읽기
long_description = (here / "README.md").read_text(encoding="utf-8")

# 버전 정보 읽기 (간단한 방법)
version = "0.1.0"

# CUDA 파일 찾기
def find_cuda_files():
    cuda_files = []
    gpu_dir = here / "gpe_core" / "gpu"
    if gpu_dir.exists():
        for ext in ["*.cu", "*.cuh"]:
            cuda_files.extend([str(f.relative_to(here)) for f in gpu_dir.glob(ext)])
    return cuda_files

setup(
    name="gpe-core",
    version=version,
    author="YC Math",
    author_email="your-email@example.com",
    description="Core library for Generative Payload Encapsulation (GPE) protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ycmath/gpe_core",
    project_urls={
        "Bug Tracker": "https://github.com/ycmath/gpe_core/issues",
        "Documentation": "https://github.com/ycmath/gpe_core/wiki",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="gpe data-exchange encoding gpu acceleration",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    python_requires=">=3.8",
    
    # 기본 의존성
    install_requires=[
        "numpy>=1.21.0",
        "orjson>=3.8.0",
        "xxhash>=3.0.0",
    ],
    
    # 선택적 의존성
    extras_require={
        "numba": ["numba>=0.56.0"],
        "gpu": [
            "cupy-cuda11x>=11.0.0",  # CUDA 11.x 버전
            # 또는 "cupy-cuda12x>=11.0.0",  # CUDA 12.x 버전
        ],
        "gpu-multi": [
            "cupy-cuda11x>=11.0.0",
            "ray[default]>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "tqdm>=4.65.0",
        ],
        "all": [
            "numba>=0.56.0",
            "cupy-cuda11x>=11.0.0",
            "ray[default]>=2.0.0",
            "tqdm>=4.65.0",
        ],
    },
    
    # CLI 엔트리 포인트
    entry_points={
        "console_scripts": [
            "gpe=gpe_core.cli:main",
        ],
    },
    
    # 패키지 데이터 포함
    package_data={
        "gpe_core": ["py.typed"],  # 타입 힌트 지원
        "gpe_core.gpu": ["*.cu", "*.cuh"],  # CUDA 소스 파일
    },
    include_package_data=True,
    zip_safe=False,  # CUDA 파일 때문에 압축 불가
)

# 설치 도움말
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPE Core 설치 옵션:")
    print("="*60)
    print("기본 설치:           pip install .")
    print("Numba 가속:          pip install .[numba]")
    print("GPU 지원:            pip install .[gpu]")
    print("멀티 GPU:            pip install .[gpu-multi]")
    print("개발 도구:           pip install .[dev]")
    print("모든 기능:           pip install .[all]")
    print("="*60 + "\n")
