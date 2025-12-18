"""
增强的配置管理器，提供类似 MaiBot 主配置的更新机制
功能：
1. 备份配置文件到 old 目录
2. 智能合并新旧配置
3. 版本检测和自动更新
"""

import os
import shutil
import datetime
from typing import Dict, Any, Optional
import toml
import json


class EnhancedConfigManager:
    """增强的配置管理器，提供类似 MaiBot 主配置的更新机制"""
    
    def __init__(self, plugin_dir: str, config_file_name: str = "config.toml"):
        """
        初始化配置管理器
        
        Args:
            plugin_dir: 插件目录路径
            config_file_name: 配置文件名
        """
        self.plugin_dir = plugin_dir
        self.config_file_name = config_file_name
        self.config_file_path = os.path.join(plugin_dir, config_file_name)
        self.old_dir = os.path.join(plugin_dir, "old")
        
        # 创建 old 目录
        os.makedirs(self.old_dir, exist_ok=True)
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """
        清理旧的备份文件，只保留最新的 keep_count 个
        
        Args:
            keep_count: 要保留的备份文件数量
        """
        try:
            # 列出 old 目录中所有以 .auto_backup_ 开头并以 .toml 结尾的文件
            import glob
            pattern = os.path.join(self.old_dir, f"{self.config_file_name}.auto_backup_*.toml")
            backup_files = glob.glob(pattern)
            # 按修改时间降序排序，如果修改时间相同则按文件名降序排序（确保最早的文件在最后）
            backup_files.sort(key=lambda f: (os.path.getmtime(f), os.path.basename(f)), reverse=True)
            # 删除超出保留数量的文件
            for file_path in backup_files[keep_count:]:
                try:
                    os.remove(file_path)
                    print(f"[EnhancedConfigManager] 删除旧备份文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[EnhancedConfigManager] 删除备份文件失败 {file_path}: {e}")
        except Exception as e:
            print(f"[EnhancedConfigManager] 清理备份文件时出错: {e}")
    
    def backup_config(self, version: str = "") -> str:
        """
        备份配置文件到 old 目录
        
        Args:
            version: 配置版本号，用于文件名
            
        Returns:
            str: 备份文件路径，如果失败则返回空字符串
        """
        print(f"[EnhancedConfigManager] 尝试备份配置文件，版本={version}")
        print(f"[EnhancedConfigManager] 配置文件路径: {self.config_file_path}")
        print(f"[EnhancedConfigManager] old 目录: {self.old_dir}")
        if not os.path.exists(self.config_file_path):
            print(f"[EnhancedConfigManager] 配置文件不存在，跳过备份")
            return ""
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_suffix = f"_v{version}" if version else ""
        # 自动备份文件名添加 "auto" 标记，并以 .toml 结尾
        backup_name = f"{self.config_file_name}.auto_backup_{timestamp}{version_suffix}.toml"
        backup_path = os.path.join(self.old_dir, backup_name)
        print(f"[EnhancedConfigManager] 备份文件名: {backup_name}")
        
        try:
            shutil.copy2(self.config_file_path, backup_path)
            # 更新备份文件的修改时间为当前时间，确保在清理时它被视为最新的
            os.utime(backup_path, None)  # None 表示设置为当前时间
            print(f"[EnhancedConfigManager] 备份成功: {backup_path}")
            # 备份成功后清理旧备份，保留10个
            self._cleanup_old_backups(keep_count=10)
            return backup_path
        except Exception as e:
            print(f"[EnhancedConfigManager] 备份配置文件失败: {e}")
            return ""
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典，如果文件不存在或解析失败则返回空字典
        """
        if not os.path.exists(self.config_file_path):
            return {}
        
        try:
            with open(self.config_file_path, "r", encoding="utf-8") as f:
                return toml.load(f) or {}
        except Exception as e:
            print(f"[EnhancedConfigManager] 加载配置文件失败: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """
        保存配置文件
        
        Args:
            config: 配置字典
        """
        try:
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                toml.dump(config, f)
        except Exception as e:
            print(f"[EnhancedConfigManager] 保存配置文件失败: {e}")
    
    def _format_toml_value(self, value: Any) -> str:
        """将Python值格式化为合法的TOML字符串（用于注释生成）"""
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            inner = ", ".join(self._format_toml_value(item) for item in value)
            return f"[{inner}]"
        if isinstance(value, dict):
            items = [f"{k} = {self._format_toml_value(v)}" for k, v in value.items()]
            return "{ " + ", ".join(items) + " }"
        return json.dumps(value, ensure_ascii=False)
    
    def save_config_with_comments(self, config: Dict[str, Any], schema: Dict[str, Any]):
        """
        保存配置文件并保留注释（基于schema）
        保留所有配置节，即使不在schema中
        支持嵌套子表（如 models.model1）和动态模型配置（如 model2）
        避免在父节中输出子字典作为内联表，以防止TOML解析错误。
        
        Args:
            config: 配置字典
            schema: 配置schema，用于生成注释
        """
        try:
            toml_str = f"# {self.config_file_name} - 配置文件\n"
            toml_str += f"# 自动生成于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # 辅助函数：从嵌套字典获取点分隔节的值
            def get_nested_section(config, section):
                parts = section.split('.')
                cur = config
                for part in parts:
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        return {}
                return cur if isinstance(cur, dict) else {}
            
            # 收集所有节：config中的节和schema中的节的并集
            all_sections = set(config.keys()) | set(schema.keys())
            # 同时，需要识别config中的嵌套子表（例如 models.model1 可能不在顶级键中）
            # 我们递归遍历config，收集所有点分隔的路径
            def collect_sections(d, prefix=""):
                sections = set()
                for key, value in d.items():
                    if isinstance(value, dict):
                        full_key = f"{prefix}.{key}" if prefix else key
                        sections.add(full_key)
                        sections.update(collect_sections(value, full_key))
                return sections
            nested_sections = collect_sections(config)
            all_sections.update(nested_sections)
            
            # 构建子节映射：对于每个节，找出其直接子节
            child_map = {}
            for section in all_sections:
                parts = section.split('.')
                if len(parts) > 1:
                    parent = '.'.join(parts[:-1])
                    child_map.setdefault(parent, set()).add(section)
            
            # 辅助函数：判断一个字段是否应被跳过（因为它是子节）
            def should_skip_field(section, field_name, value):
                # 如果字段值不是字典，不跳过
                if not isinstance(value, dict):
                    return False
                # 构造完整路径
                if section:
                    full_path = f"{section}.{field_name}"
                else:
                    full_path = field_name
                # 如果完整路径在 all_sections 中，说明有专门的子节，跳过
                return full_path in all_sections
            
            # 按照schema中定义的顺序对节进行排序
            schema_sections = [s for s in schema.keys() if s in all_sections]
            extra_sections = [s for s in all_sections if s not in schema]
            
            # 保持schema中的顺序（Python 3.7+ 字典保持插入顺序）
            ordered_sections = []
            for section in schema.keys():
                if section in all_sections:
                    ordered_sections.append(section)
            # 剩余的节按字母顺序排序
            ordered_sections.extend(sorted(extra_sections))
            
            # 处理所有节
            for section in ordered_sections:
                fields = schema.get(section) if section in schema else None
                
                # 获取该节的配置值（可能来自嵌套）
                section_config = get_nested_section(config, section)
                if not section_config:
                    # 如果嵌套获取失败，尝试顶级键
                    section_config = config.get(section, {})
                
                # 如果节配置为空，跳过
                if not section_config:
                    continue
                
                # 添加节标题
                toml_str += f"[{section}]\n\n"
                
                if fields and isinstance(fields, dict):
                    # 处理schema中定义的字段（带注释）
                    for field_name, field_info in fields.items():
                        if "description" in field_info:
                            toml_str += f"# {field_info['description']}\n"
                        
                        # 获取字段值：优先使用配置中的值，否则使用默认值
                        value = section_config.get(field_name, field_info.get("default", ""))
                        toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"
                    
                    # 对于schema中未定义但配置中存在的字段，也输出（不带注释）
                    for field_name, value in section_config.items():
                        if field_name in fields:
                            continue  # 已经处理过
                        # 如果是子节，跳过
                        if should_skip_field(section, field_name, value):
                            continue
                        toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"
                else:
                    # 不在schema中的节，输出所有字段（不带注释）
                    for field_name, value in section_config.items():
                        if should_skip_field(section, field_name, value):
                            continue
                        toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"
            
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                f.write(toml_str)
        except Exception as e:
            print(f"[EnhancedConfigManager] 保存带注释的配置文件失败: {e}")
            # 回退到普通保存
            self.save_config(config)
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将点分隔的键转换为嵌套字典。
        例如 {'models.model1': {...}} 转换为 {'models': {'model1': {...}}}
        同时保留其他键。
        """
        normalized = {}
        for key, value in config.items():
            if '.' in key:
                parts = key.split('.')
                current = normalized
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        # 最后一部分，赋值
                        if isinstance(current, dict):
                            current[part] = value
                    else:
                        if part not in current:
                            current[part] = {}
                        elif not isinstance(current[part], dict):
                            # 冲突，转换为字典
                            current[part] = {}
                        current = current[part]
            else:
                normalized[key] = value
        return normalized

    def merge_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并新旧配置，保留用户自定义值
        
        算法类似 MaiBot 的 _update_dict 函数：
        1. 以新配置为基准
        2. 将旧配置的值合并到新配置中
        3. 跳过 version/config_version 字段
        4. 递归处理嵌套字典
        
        同时处理点分隔的键（如 models.model1）与嵌套字典的转换，确保结构一致。
        
        Args:
            old_config: 旧配置（用户的自定义配置）
            new_config: 新配置（默认配置模板）
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        # 规范化新配置（可能包含点分隔键）
        norm_new = self._normalize_config(new_config)
        # 旧配置可能已经是嵌套结构，但也可能包含点分隔键（不太可能），同样规范化
        norm_old = self._normalize_config(old_config)
        
        def _merge_dicts(base: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
            """递归合并字典，保留用户自定义值"""
            result = base.copy()
            
            for key, user_value in user.items():
                # 跳过版本字段
                if key in ["version", "config_version"]:
                    continue
                    
                if key in result:
                    base_value = result[key]
                    if isinstance(user_value, dict) and isinstance(base_value, dict):
                        # 递归合并嵌套字典
                        result[key] = _merge_dicts(base_value, user_value)
                    else:
                        # 保留用户的自定义值
                        result[key] = user_value
                else:
                    # 旧配置中有但新配置中没有的键，保留但记录
                    result[key] = user_value
                    print(f"[EnhancedConfigManager] 保留已移除的配置项: {key}")
            
            return result
        
        merged = _merge_dicts(norm_new, norm_old)
        return merged
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """
        比较两个版本号
        
        Args:
            version1: 第一个版本号
            version2: 第二个版本号
            
        Returns:
            int: -1 如果 version1 < version2
                 0 如果 version1 == version2
                 1 如果 version1 > version2
        """
        def _v_to_tuple(v):
            # 移除 'v' 前缀并分割
            v = v.lstrip('v')
            # 分割主版本、次版本、修订版本
            parts = v.split('.')
            # 转换为整数，忽略非数字部分
            result = []
            for part in parts:
                # 提取数字部分
                num = ''
                for ch in part:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                result.append(int(num) if num else 0)
            # 补齐到3位
            while len(result) < 3:
                result.append(0)
            return tuple(result[:3])
        
        v1_tuple = _v_to_tuple(version1)
        v2_tuple = _v_to_tuple(version2)
        
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    
    def get_config_version(self, config: Dict[str, Any]) -> str:
        """
        获取配置版本号
        
        Args:
            config: 配置字典
            
        Returns:
            str: 版本号，如果没有则返回 "0.0.0"
        """
        if "plugin" in config and "config_version" in config["plugin"]:
            return str(config["plugin"]["config_version"])
        return "0.0.0"
    
    def compare_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较新旧配置，生成变更报告
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            
        Returns:
            Dict[str, Any]: 变更报告，包含新增、删除、修改的配置项
        """
        # 规范化配置以确保结构一致
        norm_old = self._normalize_config(old_config)
        norm_new = self._normalize_config(new_config)
        
        changes = {
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": []
        }
        
        def _compare_dicts(old: Dict[str, Any], new: Dict[str, Any], path: str = ""):
            """递归比较字典"""
            all_keys = set(old.keys()) | set(new.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key in ["version", "config_version"]:
                    continue
                    
                if key not in old:
                    # 新增的键
                    changes["added"].append(current_path)
                elif key not in new:
                    # 删除的键
                    changes["removed"].append(current_path)
                else:
                    old_value = old[key]
                    new_value = new[key]
                    
                    if isinstance(old_value, dict) and isinstance(new_value, dict):
                        _compare_dicts(old_value, new_value, current_path)
                    elif old_value != new_value:
                        # 值被修改
                        changes["modified"].append({
                            "path": current_path,
                            "old": old_value,
                            "new": new_value
                        })
                    else:
                        changes["unchanged"].append(current_path)
        
        _compare_dicts(norm_old, norm_new)
        return changes
    
    def update_config_if_needed(
        self,
        expected_version: str,
        default_config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        old_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        检查并更新配置（如果需要）
        
        Args:
            expected_version: 期望的配置版本
            default_config: 默认配置结构（来自schema）
            schema: 配置schema，用于生成带注释的配置文件
            old_config: 可选的旧配置字典。如果提供，将使用此配置而不是从文件加载。
            
        Returns:
            Dict[str, Any]: 更新后的配置
        """
        print(f"[EnhancedConfigManager] 开始检查配置更新，期望版本 v{expected_version}")
        
        # 加载现有配置
        if old_config is None:
            print(f"[EnhancedConfigManager] 从文件加载配置: {self.config_file_path}")
            old_config = self.load_config()
        else:
            print(f"[EnhancedConfigManager] 使用提供的旧配置（跳过文件加载）")
        
        # 如果配置文件不存在，使用默认配置
        if not old_config:
            print(f"[EnhancedConfigManager] 配置文件不存在，使用默认配置 v{expected_version}")
            final_config = default_config
            if schema:
                print(f"[EnhancedConfigManager] 保存带注释的默认配置")
                self.save_config_with_comments(final_config, schema)
            else:
                print(f"[EnhancedConfigManager] 保存默认配置")
                self.save_config(final_config)
            return final_config
        
        current_version = self.get_config_version(old_config)
        print(f"[EnhancedConfigManager] 当前配置版本 v{current_version}, 期望版本 v{expected_version}")
        
        # 如果版本相同，不需要更新
        if current_version == expected_version:
            print(f"[EnhancedConfigManager] 配置版本已是最新 v{current_version}")
            return old_config
        
        # 版本不同，无论高低都先备份当前配置文件
        print(f"[EnhancedConfigManager] 版本不同，开始备份当前配置")
        backup_path = self.backup_config(current_version)
        if backup_path:
            print(f"[EnhancedConfigManager] 已备份旧配置文件到: {backup_path}")
        else:
            print(f"[EnhancedConfigManager] 备份失败或配置文件不存在")
        
        print(f"[EnhancedConfigManager] 检测到配置版本需要更新: 当前=v{current_version}, 期望=v{expected_version}")
        
        # 比较配置变化
        print(f"[EnhancedConfigManager] 开始比较新旧配置差异")
        changes = self.compare_configs(old_config, default_config)
        if changes["added"]:
            print(f"[EnhancedConfigManager] 新增配置项: {', '.join(changes['added'])}")
        if changes["removed"]:
            print(f"[EnhancedConfigManager] 移除配置项: {', '.join(changes['removed'])}")
        if changes["modified"]:
            for mod in changes["modified"]:
                path = mod['path']
                old_val = mod['old']
                new_val = mod['new']
                # 隐藏敏感字段的值
                if any(sensitive in path.lower() for sensitive in ['api_key', 'key', 'token', 'secret', 'password']):
                    old_val = '***' if old_val else ''
                    new_val = '***' if new_val else ''
                print(f"[EnhancedConfigManager] 修改配置项: {path} (旧值: {old_val} -> 新值: {new_val})")
        if not changes["added"] and not changes["removed"] and not changes["modified"]:
            print(f"[EnhancedConfigManager] 配置内容无变化（仅版本号不同）")
        
        # 合并配置
        print(f"[EnhancedConfigManager] 开始合并新旧配置")
        merged_config = self.merge_configs(old_config, default_config)
        print(f"[EnhancedConfigManager] 合并完成")
        
        # 调试：检查 api_key 是否保留（不打印具体值）
        def get_nested_value(config, path):
            parts = path.split('.')
            cur = config
            for part in parts:
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur
        api_key_value = get_nested_value(merged_config, "models.model1.api_key")
        if api_key_value:
            print(f"[EnhancedConfigManager] 合并后 api_key 已保留（长度: {len(api_key_value)}）")
        else:
            print(f"[EnhancedConfigManager] 警告: 合并后未找到 api_key")
        
        # 更新版本号
        if "plugin" in merged_config:
            merged_config["plugin"]["config_version"] = expected_version
            print(f"[EnhancedConfigManager] 更新配置版本号 -> v{expected_version}")
        
        # 保存新配置
        if schema:
            print(f"[EnhancedConfigManager] 保存带注释的配置文件")
            self.save_config_with_comments(merged_config, schema)
        else:
            print(f"[EnhancedConfigManager] 保存配置文件")
            self.save_config(merged_config)
        
        print(f"[EnhancedConfigManager] 配置文件已从 v{current_version} 更新到 v{expected_version}")
        
        return merged_config