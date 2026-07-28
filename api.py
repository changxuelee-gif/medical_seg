"""
医疗影像病灶分割API - 预留部署接口

这是一个基于FastAPI的医疗影像分割服务预留接口，用于将训练好的U-Net/UNet++模型
部署为HTTP服务，支持前端或其他系统通过REST API调用进行病灶分割。

==================== 启动方式 ====================
方式一：使用uvicorn命令行启动
    uvicorn api:app --host 0.0.0.0 --port 8000

方式二：直接运行本文件
    python api.py

==================== 接口文档 ====================
启动后访问自动生成的交互式API文档：
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc

==================== 可扩展功能 ====================
本接口为预留演示版本，生产环境可进一步扩展：
1. 用户认证（API Key、JWT、OAuth2等）
2. 批量图片处理接口
3. 异步任务处理（Celery + Redis/RabbitMQ）
4. 请求限流（Rate Limiting）
5. 日志记录与监控（Prometheus、Grafana）
6. 模型版本管理与A/B测试
7. DICOM格式医学影像支持
8. 结果存储（对象存储、数据库）
"""

import sys
from pathlib import Path
from io import BytesIO

import base64
import json

import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from medicalseg.inference import Predictor
from medicalseg.utils.device import get_device

app = FastAPI(
    title="医疗影像病灶分割API",
    description="基于U-Net/UNet++的医疗影像病灶分割RESTful接口",
    version="1.0.0",
)

predictor: Predictor | None = None

_checkpoint_path = _project_root / "checkpoints" / "best_model.pth"


def _create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    创建原始图像与掩码的叠加可视化图

    将二值掩码以指定颜色半透明叠加到原始灰度图像上，便于直观查看
    病灶在原始影像中的位置和形状。

    使用简单的alpha混合公式：
    result = background * (1 - alpha) + color * alpha
    其中background是原始灰度图像（已转为RGB），color是叠加颜色。

    Args:
        original: 原始图像数组，形状(H,W)，值域[0,1]
        mask: 二值掩码数组，形状(H,W)，值域{0,1}
        alpha: 叠加透明度，0-1之间，默认0.5
        color: 叠加颜色RGB元组，默认红色(255,0,0)

    Returns:
        叠加后的RGB图像数组，形状(H,W,3)，值域[0,255]，uint8类型
    """
    original_uint8 = (original * 255).astype(np.uint8)
    original_rgb = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32)

    overlay = original_rgb.copy()
    color_r = float(color[0])
    color_g = float(color[1])
    color_b = float(color[2])

    mask_bool = mask.astype(bool)

    for c in range(3):
        if c == 0:
            channel_color = color_r
        elif c == 1:
            channel_color = color_g
        else:
            channel_color = color_b
        overlay[:, :, c][mask_bool] = overlay[:, :, c][mask_bool] * (1 - alpha) + channel_color * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def _encode_image_to_base64(image_array: np.ndarray, mode: str = "L") -> str:
    """
    将numpy数组图像编码为base64 PNG字符串

    Args:
        image_array: 图像numpy数组
            - 灰度图: 形状(H,W)，值域[0,1]或[0,255]
            - RGB图: 形状(H,W,3)，值域[0,255]
        mode: PIL图像模式，"L"为灰度，"RGB"为彩色

    Returns:
        base64编码的PNG图像字符串，可直接用于前端img标签src属性
    """
    if image_array.max() <= 1.0 and mode == "L":
        image_array = (image_array * 255).astype(np.uint8)
    elif image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    pil_image = Image.fromarray(image_array, mode=mode)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


@app.on_event("startup")
async def load_model():
    """
    应用启动时加载模型

    FastAPI启动事件钩子，在服务启动时自动尝试加载训练好的模型检查点。
    如果检查点文件不存在，打印警告信息但不崩溃，服务仍可启动，
    只是/predict接口会返回503错误提示需要先训练模型。
    """
    global predictor

    try:
        device = get_device(verbose=True)
    except Exception as e:
        print(f"[警告] 设备检测失败，使用CPU: {e}")
        import torch
        device = torch.device("cpu")

    if _checkpoint_path.exists():
        try:
            print(f"[启动] 正在加载模型检查点: {_checkpoint_path}")
            predictor = Predictor(checkpoint_path=_checkpoint_path, device=device)
            print(f"[启动] 模型加载成功，设备: {device}")
        except Exception as e:
            print(f"[警告] 模型加载失败: {e}")
            predictor = None
    else:
        print(f"[警告] 模型检查点文件不存在: {_checkpoint_path}")
        print(f"[警告] 请先训练模型，或检查路径是否正确")
        print(f"[警告] /predict 接口将返回503错误，直到模型加载成功")
        predictor = None


@app.get("/health", summary="健康检查", description="检查服务运行状态和模型是否已加载")
async def health_check():
    """
    健康检查接口

    用于监控服务是否正常运行，以及模型是否已成功加载。
    通常被负载均衡器、Kubernetes liveness/readiness探针调用。

    Returns:
        JSON对象包含:
        - status: 服务状态，"ok"表示正常
        - model_loaded: 布尔值，模型是否已加载
        - device: 当前使用的计算设备（cuda:0 或 cpu）
    """
    device_str = str(predictor.device) if predictor is not None else "cpu"
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": device_str,
    }


@app.post(
    "/predict",
    summary="影像分割预测",
    description="上传医疗影像图片，返回分割掩码和叠加可视化结果",
)
async def predict(file: UploadFile = File(..., description="待分割的医疗影像文件（支持PNG/JPG等格式）")):
    """
    医学影像分割预测接口

    接收上传的图片文件，执行以下流程：
    1. 读取上传的图片内容
    2. 转换为灰度numpy数组
    3. 调用模型进行病灶分割
    4. 将原始图、掩码、叠加图编码为base64返回

    Args:
        file: UploadFile类型，上传的图片文件

    Returns:
        JSON对象包含:
        - success: 布尔值，是否成功
        - original: 原始图像base64编码字符串
        - mask: 预测掩码base64编码字符串（白色为病灶区域）
        - overlay: 红色叠加可视化图base64编码字符串
        - threshold: 二值化阈值，默认0.5

    Raises:
        HTTPException 503: 模型未加载（需要先训练模型）
        HTTPException 500: 预测过程中发生其他错误
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请先训练模型并确保checkpoints/best_model.pth存在",
        )

    try:
        contents = await file.read()

        try:
            pil_image = Image.open(BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析图片文件: {str(e)}")

        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")

        image_np = np.array(pil_image)
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.float32) / 255.0

        threshold = 0.5
        original_img, pred_mask, prob_map = predictor.predict(image_np, threshold=threshold)

        overlay_img = _create_overlay(original_img, pred_mask)
        mask_uint8 = (pred_mask * 255).astype(np.uint8)

        original_b64 = _encode_image_to_base64(original_img, mode="L")
        mask_b64 = _encode_image_to_base64(mask_uint8, mode="L")
        overlay_b64 = _encode_image_to_base64(overlay_img, mode="RGB")

        return JSONResponse(
            content={
                "success": True,
                "original": original_b64,
                "mask": mask_b64,
                "overlay": overlay_b64,
                "threshold": threshold,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"预测失败: {str(e)}\n{traceback.format_exc()}"
        print(f"[错误] {error_detail}")
        raise HTTPException(status_code=500, detail=f"预测过程发生错误: {str(e)}")


if __name__ == "__main__":
    """
    直接运行本文件时的入口

    使用uvicorn启动FastAPI服务，监听0.0.0.0:8000。
    uvicorn是可选依赖，如果未安装会给出提示。
    """
    import argparse

    parser = argparse.ArgumentParser(description="启动医疗影像分割API服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认8000")
    parser.add_argument("--reload", action="store_true", help="开发模式，代码修改自动重载")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("=" * 60)
        print("[错误] 未安装uvicorn，请先安装：")
        print("  pip install uvicorn")
        print("")
        print("或者使用命令行方式启动：")
        print("  uvicorn api:app --host 0.0.0.0 --port 8000")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("医疗影像病灶分割API服务启动")
    print("=" * 60)
    print(f"  服务地址: http://{args.host}:{args.port}")
    print(f"  接口文档: http://localhost:{args.port}/docs")
    print(f"  ReDoc文档: http://localhost:{args.port}/redoc")
    print(f"  健康检查: http://localhost:{args.port}/health")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
