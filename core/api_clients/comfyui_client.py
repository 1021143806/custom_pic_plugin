"""ComfyUI API客户端

支持本地或远程的ComfyUI服务器，通过工作流JSON执行图片生成
"""
import json
import time
import base64
import requests
from typing import Dict, Any, Tuple, Optional

from .base_client import BaseApiClient, logger
from ..size_utils import parse_pixel_size


class ComfyUIClient(BaseApiClient):
    """ComfyUI API客户端"""

    format_name = "comfyui"

    def _make_request(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        size: str,
        strength: float = None,
        input_image_base64: str = None
    ) -> Tuple[bool, str]:
        """发送ComfyUI格式的HTTP请求生成图片"""
        try:
            # API配置
            server_url = model_config.get("base_url", "http://127.0.0.1:8188").rstrip('/')
            api_key = model_config.get("api_key", "")

            # 获取工作流模板
            workflow_template = model_config.get("workflow", {})
            if not workflow_template:
                logger.error(f"{self.log_prefix} (ComfyUI) 未配置工作流模板")
                return False, "未配置ComfyUI工作流模板"

            # 获取模型特定的配置参数
            custom_prompt_add = model_config.get("custom_prompt_add", "")
            negative_prompt = model_config.get("negative_prompt_add", "")
            full_prompt = prompt + custom_prompt_add

            # 解析尺寸
            width, height = self._parse_size(size, model_config)

            # 替换工作流中的变量
            workflow = self._prepare_workflow(
                workflow_template,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=model_config.get("seed", -1),
                steps=model_config.get("steps", 20),
                cfg=model_config.get("cfg", 7.0),
                input_image_base64=input_image_base64,
                strength=strength
            )

            # 构建请求payload
            payload = {
                "prompt": workflow,
            }

            # 添加API密钥（如果配置了）
            if api_key:
                payload["extra_data"] = {
                    "api_key_comfy_org": api_key
                }

            logger.info(f"{self.log_prefix} (ComfyUI) 发起图片请求: {server_url}")

            # 获取代理配置
            proxy_config = self._get_proxy_config()

            # 发送任务
            request_kwargs = {
                "url": f"{server_url}/prompt",
                "json": payload,
                "timeout": proxy_config.get('timeout', 30) if proxy_config else 30
            }

            if proxy_config:
                request_kwargs["proxies"] = {
                    "http": proxy_config["http"],
                    "https": proxy_config["https"]
                }

            response = requests.post(**request_kwargs)

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"{self.log_prefix} (ComfyUI) 提交任务失败: HTTP {response.status_code} - {error_msg}")
                return False, f"提交任务失败: {error_msg[:100]}"

            result = response.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                logger.error(f"{self.log_prefix} (ComfyUI) 未获取到任务ID")
                return False, "未获取到任务ID"

            logger.info(f"{self.log_prefix} (ComfyUI) 任务已提交，ID: {prompt_id}")

            # 轮询等待结果
            image_data = self._poll_for_result(server_url, prompt_id, proxy_config)

            if image_data:
                logger.info(f"{self.log_prefix} (ComfyUI) 图片生成成功")
                return True, image_data
            else:
                return False, "图片生成超时或失败"

        except requests.RequestException as e:
            logger.error(f"{self.log_prefix} (ComfyUI) 网络请求异常: {e}")
            return False, f"网络请求失败: {str(e)}"

        except Exception as e:
            logger.error(f"{self.log_prefix} (ComfyUI) 请求异常: {e!r}", exc_info=True)
            return False, f"请求失败: {str(e)}"

    def _parse_size(self, size: str, model_config: Dict[str, Any]) -> Tuple[int, int]:
        """解析尺寸字符串（委托给size_utils）"""
        default_width = model_config.get("default_width", 1024)
        default_height = model_config.get("default_height", 1024)
        return parse_pixel_size(size, default_width, default_height)

    def _prepare_workflow(
        self,
        template: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        input_image_base64: str = None,
        strength: float = None
    ) -> Dict[str, Any]:
        """准备工作流，替换变量

        Args:
            template: 工作流模板
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图片宽度
            height: 图片高度
            seed: 随机种子
            steps: 步数
            cfg: CFG值
            input_image_base64: 输入图片的base64编码
            strength: 图生图强度

        Returns:
            处理后的工作流字典
        """
        # 深拷贝模板
        workflow = json.loads(json.dumps(template))

        # 遍历所有节点，替换变量
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict) or "inputs" not in node_data:
                continue

            inputs = node_data["inputs"]

            # 替换提示词
            if "prompt" in inputs:
                inputs["prompt"] = prompt
            if "text" in inputs and node_data.get("class_type") in ["CLIPTextEncode", "CLIPTextEncodeSDXL"]:
                if "positive" in node_id.lower() or node_data.get("_meta", {}).get("title", "").lower().find("positive") >= 0:
                    inputs["text"] = prompt
                elif "negative" in node_id.lower() or node_data.get("_meta", {}).get("title", "").lower().find("negative") >= 0:
                    inputs["text"] = negative_prompt

            # 替换尺寸
            if "width" in inputs:
                inputs["width"] = width
            if "height" in inputs:
                inputs["height"] = height

            # 替换种子
            if "seed" in inputs:
                inputs["seed"] = seed if seed != -1 else int(time.time() * 1000) % (2**32)
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed if seed != -1 else int(time.time() * 1000) % (2**32)

            # 替换步数和CFG
            if "steps" in inputs:
                inputs["steps"] = steps
            if "cfg" in inputs:
                inputs["cfg"] = cfg

            # 替换图生图强度
            if "denoise" in inputs and strength is not None:
                inputs["denoise"] = strength

            # 处理输入图片（如果有LoadImage节点）
            if input_image_base64 and node_data.get("class_type") == "LoadImage":
                # ComfyUI需要图片文件名，这里需要先上传图片
                # 暂时跳过，后续可以实现图片上传功能
                pass

        return workflow

    def _poll_for_result(
        self,
        server_url: str,
        prompt_id: str,
        proxy_config: Optional[Dict[str, Any]],
        max_wait: int = 300
    ) -> Optional[str]:
        """轮询等待结果

        Args:
            server_url: ComfyUI服务器地址
            prompt_id: 任务ID
            proxy_config: 代理配置
            max_wait: 最大等待时间（秒）

        Returns:
            图片的base64编码，失败返回None
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # 检查任务状态
                history_kwargs = {
                    "url": f"{server_url}/history/{prompt_id}",
                    "timeout": 10
                }

                if proxy_config:
                    history_kwargs["proxies"] = {
                        "http": proxy_config["http"],
                        "https": proxy_config["https"]
                    }

                history_response = requests.get(**history_kwargs)

                if history_response.status_code == 200:
                    history = history_response.json()

                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})

                        # 查找SaveImage节点的输出
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                for image_info in node_output["images"]:
                                    filename = image_info.get("filename")
                                    subfolder = image_info.get("subfolder", "")

                                    if filename:
                                        # 获取图片
                                        image_data = self._get_image(
                                            server_url, filename, subfolder, proxy_config
                                        )
                                        if image_data:
                                            return image_data

                time.sleep(2)

            except Exception as e:
                logger.warning(f"{self.log_prefix} (ComfyUI) 轮询异常: {e}")
                time.sleep(2)

        logger.error(f"{self.log_prefix} (ComfyUI) 任务超时")
        return None

    def _get_image(
        self,
        server_url: str,
        filename: str,
        subfolder: str,
        proxy_config: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """获取生成的图片

        Args:
            server_url: ComfyUI服务器地址
            filename: 图片文件名
            subfolder: 子文件夹
            proxy_config: 代理配置

        Returns:
            图片的base64编码，失败返回None
        """
        try:
            params = {"filename": filename, "type": "output"}
            if subfolder:
                params["subfolder"] = subfolder

            image_kwargs = {
                "url": f"{server_url}/view",
                "params": params,
                "timeout": 30
            }

            if proxy_config:
                image_kwargs["proxies"] = {
                    "http": proxy_config["http"],
                    "https": proxy_config["https"]
                }

            response = requests.get(**image_kwargs)

            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')

        except Exception as e:
            logger.error(f"{self.log_prefix} (ComfyUI) 获取图片失败: {e}")

        return None
