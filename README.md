# Custom Pic Plugin - 智能多模型图片生成插件

基于 Maibot 插件的智能多模型图片生成插件，支持文生图和图生图自动识别。兼容OpenAI、豆包、Gemini、魔搭等多种API格式。提供命令式风格转换、模型配置管理、结果缓存等功能。（参考 doubao_pic_plugin 进行二次开发）

魔搭 api 的优点是调用免费，AI 绘图本身配置需求并不是很高，但是平台收费又都比较贵，魔搭社区有按天计算的免费调用限额，对应麦麦的绘图需求来说完全足够。如果想接其他风格绘图的可以使用豆包和 GPT 模型。

## 插件简介

![alt text](./md_pic/30176E6B83A79E3FB342E740564B8159.jpg)
![alt text](./md_pic/20250926-145915.png)
![alt text](./md_pic/20250926-152616.png)

## ✨ 主要特性（本插件为 MaiBot 生态下的图片生成扩展）

### 🎯 智能图片生成
   - **自动模式识别**：智能判断文生图或图生图模式
   - **LLM智能判定**：在Focus模式下使用LLM精确理解用户需求
   - **关键词触发**：在Normal模式下通过关键词快速响应

### 🛠️ 多API格式支持
   - **OpenAI格式**：兼容OpenAI、硅基流动、NewAPI等
   - **豆包格式**：火山引擎豆包专用格式
   - **Gemini格式**：Google Gemini专用格式
   - **魔搭格式**：魔搭社区专用格式

### 🎨 命令式功能
   - **风格转换**：`/pic <风格>` - 快速应用预设风格
   - **模型管理**：`/pic list`、`/pic set <模型ID>` - 动态切换模型
   - **风格管理**：`/pic styles`、`/pic style <风格名>` - 查看风格详情

### ⚙️ 高级功能
   - **动态配置**：运行时切换模型，无需重启
   - **风格别名**：支持中文别名，如"卡通"对应"cartoon"
   - **结果缓存**：相同参数自动复用结果
   - **调试开关**：可控制提示信息显示
   - **图生图开关**：模型级别的图生图支持控制

## 📋 组件说明

### Action组件 - 智能图片生成
   - **激活方式**：Focus模式使用LLM判定，Normal模式使用关键词
   - **支持场景**：
        - 文生图：用户描述要画的内容
        - 图生图：回复图片并要求修改
   - **关键词**：`画`、`绘制`、`生成图片`、`图生图`、`修改图片`等

### Command组件 - 命令式操作
1. **风格化图生图** (`/pic <风格>`)
   - 直接使用预配置的英文提示词
   - 支持风格别名（中文）
   - 需要先发送图片

2. **模型配置管理**
   - `/pic list` - 查看所有可用模型
   - `/pic set <模型ID>` - 动态切换图生图命令使用的模型
   - `/pic config` - 查看当前配置
   - `/pic reset` - 重置为默认配置

3. **风格管理**
   - `/pic styles` - 列出所有可用风格
   - `/pic style <风格名>` - 查看风格详情
   - `/pic help` - 显示帮助信息

## 🚀 快速开始

### 1. 安装插件
  - 使用命令行工具或是 git base 进入你的麦麦目录

   ```shell
   cd MaiBot/plugins
   ```

  - 克隆本仓库

   ```shell
   git clone https://github.com/1021143806/custom_pic_plugin
   ```
   
  - 重启 maibot 后你会看到在当前插件文件夹 `MaiBot/plugins/custom_pic_plugin`中生成了一个配置文件 `config.toml`
  - 按照配置文件中的说明填写必要参数后重启 MaiBot 即可让你的麦麦学会不同画风的画画（如何申请 key 请自行前往对应平台官网查看 api 文档）

### 2. 配置说明
  - 编辑 `config.toml`，配置至少一个模型：

```toml
[plugin]
enabled = true  # 是否启用插件

[models.model1]
name = "我的生图模型"  # 自定义名称（用于切换模型）
base_url = "https://api.openai.com/v1"  # 根据服务商选择填写ULR
api_key = "Bearer your_api_key_here"  # 填写你的 API 密钥（不同平台添加 key 时需要注意是否需要前缀 ‘Bearer ’。）
format = "openai"  # openai/doubao/gemini/modelscope（填写API格式，根据平台选择）
model = "dall-e-3"  # 填写你要使用的模型
support_img2img = true  # 是否开启图生图（未启用自动转为文生图）
```

### 3. 自定义参数
  - 可在 [generation] 节自定义默认模型、尺寸、指导强度、自定义提示词等参数。
        - 尺寸，可以选择让 ai 自己判断或是指定尺寸，例如 gpt-image-1 模型不支持生成 512x512 的尺寸，那么我们可以固定只生成 1024x1024 ，需要自行检查兼容性。
  - 自定义提示词建议学习相关 AI 绘图知识，提示词对模型生图影响极大，大部分生图模型与豆包模糊提示词生图不同，但是使用标准的单词与逗号组合是全模型通用的。
  - 在 models 类中配置不同的模型，model1，model2 等，可以配置不同的 api 供应商及模型。可通过配置文件中的 default_model 快速切换默认调用模型，如果没有配置，则默认为配置文件第一个模型。

## 💡 使用示例

### 自然语言生图（可以指定model1，model2 等，支持中文）
```
用户：麦麦，画一张美少女
麦麦：[生成图片]
```

### 图生图
```
用户：[发送图片]
用户：[回复 麦麦： [图片] ]，说：麦麦，把这张图的背景换成海滩
麦麦：[生成修改后的图片]
```

### 命令式风格转换
```
用户：[发送图片]
用户：[回复 麦麦： [图片] ]，说：/pic cartoon
麦麦：[应用卡通风格]
```

### 动态切换模型
```
用户：/pic list
麦麦：📋 可用模型列表：
• model1 ✅[默认] 🔧[命令] 🖼️[图生图]
• model2 📝[仅文生图]

用户：/pic set model2
麦麦：✅ 图生图命令模型已成功切换！
```

## ⚙️ 配置说明

### 基础配置
- `plugin.enabled` - 是否启用插件
- `generation.default_model` - Action组件使用的默认模型
- `components.pic_command_model` - Command组件使用的模型

### 高级配置
- `components.enable_debug_info` - 调试信息开关
- `cache.enabled` - 结果缓存开关
- `cache.max_size` - 最大缓存数量

### 风格配置
```toml
[styles]
cartoon = "cartoon style, anime style, colorful, vibrant colors"
oil_painting = "oil painting style, classic art, brush strokes"

[style_aliases]
cartoon = "卡通,动漫"
oil_painting = "油画,古典"
```

### 模型配置
每个模型支持独立配置：
- `support_img2img` - 是否支持图生图
- `fixed_size_enabled` - 是否固定图片尺寸
- `guidance_scale` - 指导强度
- `num_inference_steps` - 推理步数
- `custom_prompt_add` - 正面提示词增强
- `negative_prompt_add` - 负面提示词

## 🔧 依赖说明

- 需 Python 3.12+
- 依赖 MaiBot 插件系统（0.8.0 新插件系统，测试兼容 0.10.0 - 0.10.2）
- 火山方舟 api 需要通过 pip install 'volcengine-python-sdk[ark]' 安装方舟SDK

## 常见问题

- **API 密钥未配置/错误**：请检查 `config.toml` 中 `[api] volcano_generate_api_key`。
- **图片描述为空**：需提供明确的图片描述。
- **图片尺寸无效**：支持如 `1024x1024`，宽高范围 100~10000。
- **依赖缺失**：请确保 MaiBot 插件系统相关依赖已安装。
- **api 调用报错**
400：参数不正确，请参考报错信息（message）修正不合法的请求参数，可能为插件发送的报文不兼容对应 api 供应商；
401：API Key 没有正确设置；
403：权限不够，最常见的原因是该模型需要实名认证，其他情况参考报错信息（message）；
429：触发了 rate limits；参考报错信息（message）判断触发的是 RPM /RPD / TPM / TPD / IPM / IPD 中的具体哪一种，可以参考 Rate Limits 了解具体的限流策略
504 / 503：一般是服务系统负载比较高，可以稍后尝试；

## 魔搭链接及教程

==具体流程步骤如下：==

1. 注册一个魔搭账号。
2. 然后你需要根据魔搭[官网阿里云绑定教程](https://modelscope.cn/docs/accounts/aliyun-binding-and-authorization)完成阿里云认证。
3. 接着到你的魔搭主页申请一个 API key，参考[API推理介绍](https://modelscope.cn/docs/model-service/API-Inference/intro)。
4. 现在你已经拥有了一个 key 可以直接去[模型库](https://modelscope.cn/models)挑选你想要使用的生图模型了，在每个模型的详细里都会有一段教程告诉你怎么使用，我们只需要取可以使用 API 推理的模型的模型名称就好了。
5. 在该插件的配置文件中填入你获取的 key ，选择魔搭对应的请求地址，然后填入对应的模型名称即可。剩下的相关配置根据配置文件中的注释填入。

## 未来计划

考虑兼容 Comfyui 实现自定义生图。

## 📝 更新日志

### v3.1.2
- 🎯 智能文生图/图生图自动识别
- 🛠️ 新增命令式配置管理功能
- 🎨 风格别名系统
- ⚡ 动态模型切换
- 🐛 修复失败缓存共享问题
- 🔧 优化API路径处理
- 📋 简化显示信息

### v3.1.1
- 支持多模型配置
- 新增缓存机制
- 兼容多种API格式

## 🤝 基于 MaiBot 项目
- 支持 0.8.x - 0.10.x
  - 0.9.x 升级仅配置文件新增两个字段，所以不影响 0.8 版本使用，
  - 0.10 修改支持版本号可直接加载成功
  - 目前改为一直支持最新版

插件开发历程

- 该插件基于 MaiBot 最早期官方豆包生图示例插件修改而来，最早我是为了兼容 GPT 生图进行修改，添加对 GPT 生图模型直接返回 base64 格式图片的兼容判断，因为 GPT 生图太贵了，所以后续想兼容魔搭社区的免费生图，新增一层报文兼容。（我不是计算机专业，大部分代码来自 DeepSeek R1 研究了很久，不得不说确实很好玩。）
- 目前支持三种报文返回，即三个平台的图片返回报文 url，image，base64，如果其他平台返回的报文符合以上三种格式也可以正常使用，可以自行尝试。
- MaiBot 0.8 版本更新，根据新插件系统进行重构。
- Rabbit-Jia-Er 加入，添加可以调用多个模型和命令功能。

## 🔗 版权信息

- 作者：MaiBot 团队
- 许可证：GPL-v3.0-or-later
- 项目主页：https://github.com/MaiM-with-u/maibot

---

## 📸 效果展示

![效果图1](./md_pic/70F9287538F77AC42696F002866C16BA.png)
![效果图2](./md_pic/treemodel1.png)