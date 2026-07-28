"""
医疗影像病灶分割系统 - Gradio Web界面

提供基于深度学习的医疗影像2D病灶分割Web交互界面，
支持上传图片、调整分割阈值、实时查看分割结果。
适配 Gradio 6.x 版本
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from medicalseg.inference import Predictor
from medicalseg.utils.device import get_device
from medicalseg.visualization import setup_chinese_font

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
EXAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "simulated" / "images"

predictor = None
device_info = ""
model_type_info = ""
model_loaded = False


def generate_overlay_numpy(
    image: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (0.0, 1.0, 0.0)
) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image_rgb = np.stack([image, image, image], axis=-1)
    mask_binary = (pred_mask > 0.5).astype(np.float32)

    color_array = np.zeros_like(image_rgb)
    color_array[:, :, 0] = color[0]
    color_array[:, :, 1] = color[1]
    color_array[:, :, 2] = color[2]

    mask_3d = np.stack([mask_binary] * 3, axis=-1).astype(bool)

    overlay = image_rgb.copy()
    overlay[mask_3d] = image_rgb[mask_3d] * (1 - alpha) + color_array[mask_3d] * alpha

    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return overlay_uint8


def load_model():
    global predictor, device_info, model_type_info, model_loaded

    try:
        device = get_device(verbose=False)
        device_info = f"🖥️ 计算设备: {device}"

        if not DEFAULT_CHECKPOINT_PATH.exists():
            model_loaded = False
            predictor = None
            model_type_info = "⚠️ 模型未加载"
            status_msg = (
                "⚠️ **尚未找到训练好的模型**\n\n"
                "请先运行训练命令：`python tools/train.py`"
            )
            print(f"[警告] 模型检查点不存在: {DEFAULT_CHECKPOINT_PATH}")
            return status_msg, False

        print(f"[信息] 正在加载模型: {DEFAULT_CHECKPOINT_PATH}")
        predictor = Predictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=device)

        model_name = predictor.cfg.get('model', {}).get('name', 'Unknown')
        model_type_info = f"🧠 模型类型: {model_name}"
        model_loaded = True

        status_msg = (
            f"✅ **模型加载成功!**\n\n{device_info} | {model_type_info}"
        )
        print(f"[信息] 模型加载成功，设备: {device}")
        return status_msg, True

    except Exception as e:
        model_loaded = False
        predictor = None
        model_type_info = "❌ 模型加载失败"
        error_msg = str(e)
        status_msg = (
            f"❌ **模型加载失败**\n\n错误信息: {error_msg}"
        )
        print(f"[错误] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return status_msg, False


def segment_image(input_img, threshold):
    global predictor, model_loaded

    if input_img is None:
        return None, None, None, "请先上传一张医疗影像图片！"

    if not model_loaded or predictor is None:
        return None, None, None, "模型尚未加载，请先点击「🔄 重新加载模型」按钮！"

    try:
        if len(input_img.shape) == 3 and input_img.shape[2] == 3:
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = input_img.squeeze()

        gray_img_float = gray_img.astype(np.float32)
        if gray_img_float.max() > 1.0:
            gray_img_float = gray_img_float / 255.0

        original_image, pred_mask, prob_map = predictor.predict(
            image=gray_img_float,
            threshold=threshold,
            return_numpy=True
        )

        original_gray = (original_image * 255).astype(np.uint8)
        mask_uint8 = (pred_mask * 255).astype(np.uint8)
        overlay = generate_overlay_numpy(
            image=original_image,
            pred_mask=pred_mask,
            alpha=0.5,
            color=(0.0, 1.0, 0.0)
        )

        lesion_ratio = pred_mask.mean()
        result_msg = f"✅ 分割完成！阈值: {threshold:.2f}, 病灶像素占比: {lesion_ratio:.2%}"
        print(f"[信息] {result_msg}")

        return original_gray, mask_uint8, overlay, result_msg

    except Exception as e:
        error_msg = f"分割过程中发生错误: {str(e)}"
        print(f"[错误] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, None, None, error_msg


def get_example_images():
    examples = []
    if EXAMPLE_IMAGES_DIR.exists() and EXAMPLE_IMAGES_DIR.is_dir():
        image_files = sorted(EXAMPLE_IMAGES_DIR.glob("*.png"))
        image_files.extend(sorted(EXAMPLE_IMAGES_DIR.glob("*.jpg")))
        for img_path in image_files[:3]:
            examples.append([str(img_path), 0.5])
        if examples:
            print(f"[信息] 找到 {len(examples)} 张示例图片")
    return examples


def reload_model_callback():
    status_html, btn_interactive = load_model()
    info_text = f"### ℹ️ 系统信息\n{device_info}\n\n{model_type_info}"
    return status_html, btn_interactive, info_text


print("=" * 60)
print("🏥 医疗影像病灶分割系统 - 正在启动...")
print("=" * 60)

print("[信息] 正在配置中文字体...")
setup_chinese_font()

initial_status, initial_btn_interactive = load_model()
example_list = get_example_images()
initial_info_text = f"### ℹ️ 系统信息\n{device_info}\n\n{model_type_info}"


with gr.Blocks(
    title="医疗影像病灶分割系统",
) as demo:
    gr.Markdown("# 🏥 医疗影像病灶分割系统")
    gr.Markdown("基于U-Net/UNet++的深度学习医疗影像2D病灶分割 | PyTorch实现")

    status_display = gr.Markdown(initial_status)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传影像")
            input_img = gr.Image(
                type="numpy",
                label="上传医疗影像",
                sources=["upload"],
                height=400
            )

            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="分割阈值",
                info="阈值越高，判定为病灶的条件越严格"
            )

            with gr.Row():
                predict_btn = gr.Button(
                    "🔍 开始分割",
                    variant="primary",
                    interactive=initial_btn_interactive,
                    scale=2
                )
                reload_btn = gr.Button(
                    "🔄 重新加载模型",
                    variant="secondary",
                    scale=1
                )

            if example_list:
                gr.Markdown("#### 📷 示例图片（点击快速试用）")
                gr.Examples(
                    examples=example_list,
                    inputs=[input_img, threshold_slider],
                    label="示例影像",
                )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 分割结果")
            result_msg = gr.Markdown("")

            with gr.Row():
                original_out = gr.Image(
                    label="原始影像",
                    type="numpy",
                )
                mask_out = gr.Image(
                    label="病灶掩码",
                    type="numpy",
                )
                overlay_out = gr.Image(
                    label="叠加效果（绿色=病灶）",
                    type="numpy",
                )

    info_display = gr.Markdown(initial_info_text)

    predict_btn.click(
        fn=segment_image,
        inputs=[input_img, threshold_slider],
        outputs=[original_out, mask_out, overlay_out, result_msg]
    )

    reload_btn.click(
        fn=reload_model_callback,
        inputs=None,
        outputs=[status_display, predict_btn, info_display]
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Gradio Web界面启动中...")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        inbrowser=False
    )
