"""
页面 4: 文本到图像生成（Text-to-Image）
- 使用 diffusers 加载轻量文本到图像模型
- 如果环境资源不足，使用预生成样例
- 用户可调整 prompt、negative prompt、guidance scale、steps、seed
"""

import streamlit as st
import font_setup
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import io


def generate_placeholder_image(prompt: str, width: int = 256, height: int = 256):
    """
    当无法加载 diffusion 模型时，生成一个占位图像，
    显示 prompt 文本（用于演示界面布局）。
    """
    img = Image.new("RGB", (width, height), color=(30, 30, 40))

    # 绘制渐变背景
    draw = ImageDraw.Draw(img)
    for i in range(height):
        r = int(30 + (i / height) * 30)
        g = int(30 + (i / height) * 20)
        b = int(40 + (i / height) * 40)
        draw.line([(0, i), (width, i)], fill=(r, g, b))

    # 绘制文字
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 折行显示 prompt
    lines = []
    words = prompt.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) < 30:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    y_offset = height // 2 - len(lines) * 15
    draw.rectangle([20, 20, width - 20, height - 20], outline=(100, 100, 200), width=2)
    draw.text((width // 2, 30), "🧠 Text-to-Image Demo", fill=(200, 200, 255), anchor="mt", font=font)

    for line in lines:
        draw.text((width // 2, y_offset), line, fill=(255, 255, 255), anchor="mt", font=font)
        y_offset += 25

    draw.text((width // 2, height - 30), "(Diffusion Model 未加载)", fill=(150, 150, 180), anchor="mb", font=font)

    return img


def show_text2img_page():
    """Text-to-Image 页面"""
    st.title("🖼️ 文本到图像生成 (Text-to-Image)")
    st.markdown("""
    使用 **Diffusion Model（扩散模型）** 根据文本描述生成图像。
    本页面支持加载 HuggingFace Diffusers 库中的轻量模型，或展示预生成样例。

    **核心概念**：
    - **扩散过程**：逐步向图像添加噪声直至变为纯噪声
    - **逆扩散过程**：模型学习从噪声中逐步恢复图像
    - **文本引导**：通过 CLIP 等模型将文本嵌入融入去噪过程
    """)

    # 在 CPU 上运行
    import torch

    # ---- 侧边参数 ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 生成参数")

    # Prompt
    prompt = st.sidebar.text_area(
        "📝 Prompt（正向提示词）",
        value="a beautiful landscape with mountains and a lake, digital art",
        height=80,
    )
    negative_prompt = st.sidebar.text_area(
        "🚫 Negative Prompt（负向提示词）",
        value="blurry, low quality, distorted",
        height=60,
    )

    # 高级参数
    with st.sidebar.expander("🔧 高级参数", expanded=False):
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5,
                                   help="控制文本引导强度。越高越贴近 prompt，但可能丢失多样性。")
        num_steps = st.slider("Sampling Steps", 1, 50, 4, 1,
                              help="去噪步数。更多步数通常质量更好但更慢。")
        seed = st.number_input("Seed", 0, 99999, 42,
                               help="随机种子，相同 seed 产生相同结果。")
        img_size = st.selectbox("图像尺寸", [256, 384, 512], index=0,
                                help="生成图像的边长（像素）。")

    # ---- 模型加载 ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 模型加载")

    load_mode = st.sidebar.radio(
        "生成方式",
        ["☁️ HuggingFace API（在线，推荐）", "📥 本地下载模型", "💡 演示模式"],
        index=0,
        help="API 模式无需下载，直接调用 HuggingFace 云端推理。"
    )

    pipe = None       # 本地 pipeline
    api_ready = False  # API 模式是否就绪

    # ==================== API 模式 ====================
    if load_mode.startswith("☁️"):
        with st.sidebar.expander("🔑 API 设置", expanded=True):
            hf_token = st.text_input(
                "HuggingFace Token",
                type="password",
                placeholder="hf_xxxxxxxxxxxx",
                help="在 huggingface.co/settings/tokens 创建（免费）",
            )
            api_model_id = st.selectbox(
                "模型 ID",
                [
                    "black-forest-labs/FLUX.1-schnell",
                    "stabilityai/stable-diffusion-2-1",
                    "runwayml/stable-diffusion-v1-5",
                    "prompthero/openjourney",
                    "stabilityai/sdxl-turbo",
                ],
                index=0,
                help="已验证可在免费 Inference API 上使用的模型",
            )

        if hf_token:
            api_ready = True
            st.sidebar.success("✅ API 已就绪")
            st.sidebar.caption(f"当前模型: `{api_model_id}`")
        else:
            st.sidebar.warning("⚠️ 请输入 HuggingFace Token")

    # ==================== 本地下载模式 ====================
    elif load_mode.startswith("📥"):
        model_option = st.sidebar.selectbox(
            "选择模型",
            [
                "nota-ai/bk-sdm-small (轻量 ~1GB)",
                "CompVis/stable-diffusion-v1-4 (~5GB)",
                "runwayml/stable-diffusion-v1-5 (~5GB)",
            ],
            index=0,
        )
        model_id = model_option.split(" ")[0]

        load_btn = st.sidebar.button(
            "📥 加载模型", type="secondary", width="stretch",
            help="点击下载并加载（约1-5GB，仅首次）"
        )

        cache_key = f"diffusion_pipe_{model_id.replace('/', '_')}"
        if cache_key in st.session_state:
            pipe = st.session_state[cache_key]
            st.sidebar.success(f"✅ 已就绪: `{model_id}`")

        if load_btn:
            if cache_key in st.session_state:
                pipe = st.session_state[cache_key]
                st.sidebar.info("已在内存中。")
            else:
                with st.sidebar.status(f"⏳ 下载 {model_id}...", expanded=True) as load_status:
                    st.write(f"📦 `{model_id}`")
                    try:
                        from diffusers import StableDiffusionPipeline

                        @st.cache_resource(show_spinner=False)
                        def _load_sd_pipe(name):
                            p = StableDiffusionPipeline.from_pretrained(
                                name, torch_dtype=torch.float32,
                                safety_checker=None, requires_safety_checker=False,
                            )
                            p.to("cpu")
                            try:
                                p.enable_attention_slicing()
                            except Exception:
                                pass
                            return p

                        pipe = _load_sd_pipe(model_id)
                        st.session_state[cache_key] = pipe
                        load_status.update(label=f"✅ 加载成功！", state="complete", expanded=False)
                        st.sidebar.success("✅ 就绪")
                    except Exception as e:
                        err = str(e)
                        if any(k in err for k in ["401", "403"]):
                            st.sidebar.error(f"🔒 需授权")
                        elif any(k in err for k in ["Connection", "HTTPError", "MaxRetry"]):
                            st.sidebar.error("🌐 网络不通，试试 API 模式或镜像")
                        else:
                            st.sidebar.error(f"❌ {err[:120]}")
                        load_status.update(label="❌ 失败", state="error")

    # ---- 生成按钮 ----
    generate_btn = st.sidebar.button("🎨 生成图像", type="primary", width="stretch")

    # ---- 结果展示区域 ----
    st.subheader("🎨 生成结果")

    if generate_btn:
        with st.status("⏳ 正在生成图像...", expanded=True) as status:
            # ========== API 模式生成 ==========
            if load_mode.startswith("☁️") and api_ready:
                st.write(f"☁️ 调用 HuggingFace API: `{api_model_id}`...")
                start_time = time.time()

                try:
                    from huggingface_hub import InferenceClient

                    client = InferenceClient(model=api_model_id, token=hf_token, timeout=120)

                    # 调用 text_to_image
                    gen_image = client.text_to_image(
                        prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        width=img_size,
                        height=img_size,
                    )

                    elapsed = time.time() - start_time
                    status.update(
                        label=f"✅ 生成完成（{elapsed:.1f}s）",
                        state="complete", expanded=False
                    )
                    st.session_state.last_generated = gen_image

                except Exception as e:
                    err = str(e)

                    if "401" in err or "403" in err or "Authorization" in err:
                        status.update(label="🔒 Token 无效或无权限", state="error")
                        st.error(
                            "Token 无效或无权访问该模型。\n\n"
                            "**解决方法：**\n"
                            "1. 确认 Token 正确（在 huggingface.co/settings/tokens 检查）\n"
                            "2. 部分模型需在模型页面接受使用条款\n"
                            "3. 尝试切换模型"
                        )
                    elif "404" in err or "does not exist" in err.lower():
                        status.update(label="❌ 模型不可用", state="error")
                        st.error(
                            f"模型 `{api_model_id}` 在 Inference API 上不可用。\n\n"
                            "**请尝试其他模型：**\n"
                            "- `black-forest-labs/FLUX.1-schnell`\n"
                            "- `stabilityai/stable-diffusion-2-1`\n"
                            "- `prompthero/openjourney`"
                        )
                    elif "429" in err or "rate" in err.lower():
                        status.update(label="⏱️ 请求过于频繁", state="error")
                        st.warning("请求频率过高，请稍后重试。")
                    elif "timeout" in err.lower() or "timed out" in err.lower():
                        status.update(label="⏱️ 请求超时", state="error")
                        st.warning("请求超时（可能是模型冷启动），请重试。")
                    elif "Connection" in err:
                        status.update(label="🌐 网络连接失败", state="error")
                        st.error("无法连接 HuggingFace API，请检查网络。")
                    elif "loading" in err.lower():
                        status.update(label="⏳ 模型冷启动中", state="error")
                        st.warning(
                            f"🚀 模型 `{api_model_id}` 正在冷启动，请等待 30-60 秒后重试。\n\n"
                            "免费 API 的模型在空闲后会休眠，首次调用需等待启动。"
                        )
                    else:
                        status.update(label=f"❌ 生成失败", state="error")
                        st.error(f"错误: {err[:300]}")

            # ========== 本地模型生成 ==========
            elif pipe is not None:
                st.write("🔮 本地模型推理...")
                start_time = time.time()
                try:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    result = pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale, num_inference_steps=num_steps,
                        generator=generator, width=img_size, height=img_size,
                    )
                    gen_image = result.images[0]
                    elapsed = time.time() - start_time
                    status.update(label=f"✅ 完成（{elapsed:.1f}s）", state="complete", expanded=False)
                    st.session_state.last_generated = gen_image
                except Exception as e:
                    status.update(label=f"❌ 生成失败", state="error")
                    st.error(str(e)[:200])

            # ========== 演示模式 ==========
            else:
                st.write("🎨 生成演示图像...")
                time.sleep(0.3)
                gen_image = generate_placeholder_image(prompt, img_size, img_size)
                st.session_state.last_generated = gen_image
                status.update(label="✅ 演示图像已生成", state="complete", expanded=False)

    # 展示上次生成结果
    if "last_generated" in st.session_state:
        col_img, col_info = st.columns([2, 1])

        with col_img:
            st.image(st.session_state.last_generated, caption="生成结果", width="stretch")

        with col_info:
            st.markdown("### 📋 生成参数")
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | **Prompt** | {prompt[:80]}{'...' if len(prompt) > 80 else ''} |
            | **Negative** | {negative_prompt[:60]}{'...' if len(negative_prompt) > 60 else ''} |
            | **Guidance** | {guidance_scale} |
            | **Steps** | {num_steps} |
            | **Seed** | {seed} |
            | **Size** | {img_size}×{img_size} |
            """)

        # 下载按钮
        buf = io.BytesIO()
        st.session_state.last_generated.save(buf, format="PNG")
        st.download_button(
            "📥 下载图像",
            data=buf.getvalue(),
            file_name=f"generated_{seed}.png",
            mime="image/png",
        )

    else:
        st.info("👈 在左侧输入 Prompt 并点击「生成图像」按钮。")

    # ==================== 参数效果对比 ====================
    st.markdown("---")
    st.subheader("🔬 参数效果对比")

    st.markdown("""
    使用**同一 Prompt** 以不同参数生成图像，直观对比参数对结果的影响。
    当使用 API 或本地模型时，将生成真实图像；演示模式下显示占位图。
    """)

    compare_btn = st.button("🔄 生成对比图（不同 Guidance Scale）", type="secondary")

    if compare_btn:
        compare_values = [2.0, 7.5, 15.0]
        compare_labels = [
            f"Guidance = 2.0\n（低引导：多样但可能偏离）",
            f"Guidance = 7.5\n（中等引导：平衡）",
            f"Guidance = 15.0\n（高引导：贴近但缺少变化）",
        ]
        generated_comparisons = []

        # 判断是否有真实生成能力
        can_generate = (load_mode.startswith("☁️") and api_ready) or (pipe is not None)

        if can_generate:
            with st.status("⏳ 正在生成对比图像...", expanded=True) as cmp_status:
                for i, gs in enumerate(compare_values):
                    st.write(f"生成 Guidance={gs}...")
                    try:
                        if load_mode.startswith("☁️") and api_ready:
                            from huggingface_hub import InferenceClient
                            client = InferenceClient(model=api_model_id, token=hf_token, timeout=120)
                            img = client.text_to_image(
                                prompt,
                                negative_prompt=negative_prompt,
                                guidance_scale=gs,
                                num_inference_steps=num_steps,
                                width=img_size,
                                height=img_size,
                            )
                        else:
                            generator = torch.Generator(device="cpu").manual_seed(seed)
                            result = pipe(
                                prompt=prompt, negative_prompt=negative_prompt,
                                guidance_scale=gs, num_inference_steps=num_steps,
                                generator=generator, width=img_size, height=img_size,
                            )
                            img = result.images[0]
                        generated_comparisons.append(img)
                    except Exception as e:
                        st.warning(f"Guidance={gs} 生成失败: {str(e)[:80]}")
                        generated_comparisons.append(
                            generate_placeholder_image(
                                f"Failed: {prompt} (guidance={gs})", img_size, img_size
                            )
                        )
                cmp_status.update(label="✅ 对比生成完成", state="complete", expanded=False)
        else:
            # 演示模式
            for gs in compare_values:
                generated_comparisons.append(
                    generate_placeholder_image(
                        f"{prompt} (guidance={gs})", img_size, img_size
                    )
                )

        st.session_state.comparison_images = generated_comparisons
        st.session_state.comparison_labels = compare_labels

    # 展示对比结果
    if "comparison_images" in st.session_state:
        cols = st.columns(len(st.session_state.comparison_images))
        for i, col in enumerate(cols):
            with col:
                st.image(
                    st.session_state.comparison_images[i],
                    caption=st.session_state.comparison_labels[i],
                    width="stretch",
                )

    st.markdown("""
    > 💡 **观察要点**：
    > - **低 Guidance**：图像更自由发散，可能偏离 prompt
    > - **中 Guidance (7.5)**：大多数模型的默认值，平衡质量与一致性
    > - **高 Guidance**：严格贴合 prompt，但可能过度锐化、缺少自然感
    """)

    # ==================== 知识卡片 ====================
    st.markdown("---")
    st.subheader("📚 知识卡片：Diffusion Models")

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown("""
        **正向扩散 (Forward)**
        - 逐步向图像添加高斯噪声
        - 经过 T 步后变为纯噪声
        - $q(x_t|x_{t-1}) = \\mathcal{N}(\\sqrt{1-\\beta_t}x_{t-1}, \\beta_t I)$
        """)
    with col_k2:
        st.markdown("""
        **逆向扩散 (Reverse)**
        - 模型学习从噪声恢复图像
        - 使用 U-Net 预测噪声
        - $p_\\theta(x_{t-1}|x_t) = \\mathcal{N}(\\mu_\\theta(x_t,t), \\Sigma_\\theta(x_t,t))$
        """)
    with col_k3:
        st.markdown("""
        **文本引导 (CFG)**
        - Classifier-Free Guidance
        - 结合有条件与无条件预测
        - $\\hat{\\epsilon}_\\theta = \\epsilon_\\theta(x_t,\\varnothing) + w(\\epsilon_\\theta(x_t,c) - \\epsilon_\\theta(x_t,\\varnothing))$
        """)

    st.markdown("""
    > 💡 **三种生成方式**：
    > - ☁️ **API 模式（推荐）**：无需下载，输入 HuggingFace Token 即可在线生成
    > - 📥 **本地模式**：下载模型到本地，离线使用（首次 1-5GB）
    > - 💡 **演示模式**：无需任何配置，展示界面布局
    >
    > 🔑 **获取 Token**：访问 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 创建
    """)



