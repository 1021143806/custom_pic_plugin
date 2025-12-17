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
    
    def backup_config(self, version: str = "") -> str:
        """
        备份配置文件到 old 目录
        
        Args:
            version: 配置版本号，用于文件名
            
        Returns:
            str: 备份文件路径，如果失败则返回空字符串
        """
        if not os.path.exists(self.config_file_path):
            return ""
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_suffix = f"_v{version}" if version else ""
        backup_name = f"{self.config_file_name}.backup_{timestamp}{version_suffix}"
        backup_path = os.path.join(self.old_dir, backup_name)
        
        try:
            shutil.copy2(self.config_file_path, backup_path)
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
        
        Args:
            config: 配置字典
            schema: 配置schema，用于生成注释
        """
        try:
            toml_str = f"# {self.config_file_name} - 配置文件\n"
            toml_str += f"# 自动生成于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # 收集所有节：config中的节和schema中的节的并集
            all_sections = set(config.keys()) | set(schema.keys())
            
            # 先处理schema中定义的节（带注释）
            for section in sorted(all_sections):
                if section not in schema:
                    continue  # 稍后处理
                    
                fields = schema[section]
                if not isinstance(fields, dict):
                    continue
                    
                # 添加节标题
                toml_str += f"[{section}]\n\n"
                
                # 获取该节的配置值
                section_config = config.get(section, {})
                
                # 遍历schema中定义的字段
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
                    toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"
            
            # 处理不在schema中的节（不带注释）
            for section in sorted(all_sections):
                if section in schema:
                    continue  # 已经处理过
                    
                # 添加节标题
                toml_str += f"[{section}]\n\n"
                
                # 输出该节的所有字段
                section_config = config.get(section, {})
                for field_name, value in section_config.items():
                    toml_str += f"{field_name} = {self._format_toml_value(value)}\n\n"
            
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                f.write(toml_str)
        except Exception as e:
            print(f"[EnhancedConfigManager] 保存带注释的配置文件失败: {e}")
            # 回退到普通保存
            self.save_config(config)
    
    def merge_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并新旧配置，保留用户自定义值
        
        算法类似 MaiBot 的 _update_dict 函数：
        1. 以新配置为基准
        2. 将旧配置的值合并到新配置中
        3. 跳过 version/config_version 字段
        4. 递归处理嵌套字典
        
        Args:
            old_config: 旧配置
            new_config: 新配置（通常来自模板或schema）
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        def _merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
            """递归合并字典"""
            result = target.copy()
            
            for key, source_value in source.items():
                # 跳过版本字段
                if key in ["version", "config_version"]:
                    continue
                    
                if key in result:
                    target_value = result[key]
                    if isinstance(source_value, dict) and isinstance(target_value, dict):
                        result[key] = _merge_dicts(target_value, source_value)
                    else:
                        # 保留用户的自定义值（来自source）
                        result[key] = source_value
                else:
                    # 旧配置中有但新配置中没有的键，保留但记录
                    result[key] = source_value
                    print(f"[EnhancedConfigManager] 保留已移除的配置项: {key}")
            
            return result
        
        return _merge_dicts(new_config, old_config)
    
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
        
        _compare_dicts(old_config, new_config)
        return changes
    
    def update_config_if_needed(
        self, 
        expected_version: str, 
        default_config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        检查并更新配置（如果需要）
        
        Args:
            expected_version: 期望的配置版本
            default_config: 默认配置结构（来自schema）
            schema: 配置schema，用于生成带注释的配置文件
            
        Returns:
            Dict[str, Any]: 更新后的配置
        """
        # 加载现有配置
        old_config = self.load_config()
        
        # 如果配置文件不存在，使用默认配置
        if not old_config:
            print(f"[EnhancedConfigManager] 配置文件不存在，使用默认配置 v{expected_version}")
            final_config = default_config
            if schema:
                self.save_config_with_comments(final_config, schema)
            else:
                self.save_config(final_config)
            return final_config
        
        current_version = self.get_config_version(old_config)
        
        # 如果版本相同，不需要更新
        if current_version == expected_version:
            print(f"[EnhancedConfigManager] 配置版本已是最新 v{current_version}")
            return old_config
        
        print(f"[EnhancedConfigManager] 检测到配置版本需要更新: 当前=v{current_version}, 期望=v{expected_version}")
        
        # 备份旧配置
        backup_path = self.backup_config(current_version)
        if backup_path:
            print(f"[EnhancedConfigManager] 已备份旧配置文件到: {backup_path}")
        
        # 比较配置变化
        changes = self.compare_configs(old_config, default_config)
        if changes["added"]:
            print(f"[EnhancedConfigManager] 新增配置项: {', '.join(changes['added'])}")
        if changes["removed"]:
            print(f"[EnhancedConfigManager] 移除配置项: {', '.join(changes['removed'])}")
        if changes["modified"]:
            for mod in changes["modified"]:
                print(f"[EnhancedConfigManager] 修改配置项: {mod['path']} (旧值: {mod['old']} -> 新值: {mod['new']})")
        
        # 合并配置
        merged_config = self.merge_configs(old_config, default_config)
        
        # 更新版本号
        if "plugin" in merged_config:
            merged_config["plugin"]["config_version"] = expected_version
        
        # 保存新配置
        if schema:
            self.save_config_with_comments(merged_config, schema)
        else:
            self.save_config(merged_config)
        
        print(f"[EnhancedConfigManager] 配置文件已从 v{current_version} 更新到 v{expected_version}")
        
        return merged_config