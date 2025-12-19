package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Load 从 YAML 文件加载配置到任意结构体
// 调用方自己定义配置结构体，传入指针即可
func Load(path string, cfg interface{}) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	return nil
}

// LoadWithDefault 从 YAML 文件加载配置，支持多个候选路径
// 按顺序尝试加载，找到第一个存在的文件即加载
func LoadWithDefault(candidates []string, cfg interface{}) (string, error) {
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			if err := Load(path, cfg); err != nil {
				return "", err
			}
			return path, nil
		}
	}
	return "", fmt.Errorf("no config file found in candidates: %v", candidates)
}

// MustLoad 加载配置，失败则 panic
func MustLoad(path string, cfg interface{}) {
	if err := Load(path, cfg); err != nil {
		panic(err)
	}
}

// EnsureDir 确保目录存在
func EnsureDir(path string) error {
	dir := filepath.Dir(path)
	return os.MkdirAll(dir, 0755)
}

// GetEnv 获取环境变量，支持默认值
func GetEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

// Get 从配置中获取字符串值，支持点号分隔的路径（如 "logger.level"）
// 优先从环境变量获取（环境变量名为大写+下划线，如 LOGGER_LEVEL）
func Get(cfg map[string]interface{}, key string, defaultVal string) string {
	// 先检查环境变量
	envKey := strings.ToUpper(strings.ReplaceAll(key, ".", "_"))
	if val := os.Getenv(envKey); val != "" {
		return val
	}

	// 再从配置文件获取
	val := getNestedValue(cfg, key)
	if val == nil {
		return defaultVal
	}
	if s, ok := val.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", val)
}

// GetInt 从配置中获取整数值
func GetInt(cfg map[string]interface{}, key string, defaultVal int) int {
	envKey := strings.ToUpper(strings.ReplaceAll(key, ".", "_"))
	if val := os.Getenv(envKey); val != "" {
		var i int
		fmt.Sscanf(val, "%d", &i)
		return i
	}

	val := getNestedValue(cfg, key)
	if val == nil {
		return defaultVal
	}
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return defaultVal
}

// GetBool 从配置中获取布尔值
func GetBool(cfg map[string]interface{}, key string, defaultVal bool) bool {
	envKey := strings.ToUpper(strings.ReplaceAll(key, ".", "_"))
	if val := os.Getenv(envKey); val != "" {
		return val == "true" || val == "1"
	}

	val := getNestedValue(cfg, key)
	if val == nil {
		return defaultVal
	}
	if b, ok := val.(bool); ok {
		return b
	}
	return defaultVal
}

// getNestedValue 获取嵌套的配置值
func getNestedValue(cfg map[string]interface{}, key string) interface{} {
	keys := strings.Split(key, ".")
	var current interface{} = cfg

	for _, k := range keys {
		if m, ok := current.(map[string]interface{}); ok {
			current = m[k]
		} else {
			return nil
		}
	}
	return current
}
