#!/bin/bash

# Milvus-Lite 管理脚本
# 用法: ./scripts/milvus.sh {start|stop|status|log} [数据目录路径]

# 获取当前用户的 Python3 路径（避免 sudo 环境问题）
# 如果通过 sudo 运行，尝试使用原用户的 Python
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    # 优先使用用户的 Python
    if [ -x "$USER_HOME/.local/bin/python3" ]; then
        PYTHON3="$USER_HOME/.local/bin/python3"
    else
        PYTHON3=$(which python3)
    fi
else
    PYTHON3=$(which python3)
fi

# 默认数据存储路径（可通过第二个参数或环境变量 MILVUS_DATA_DIR 覆盖）
DEFAULT_DATA_DIR="/kilocai_cfs_gz/milvus/data"
DATA_DIR="${2:-${MILVUS_DATA_DIR:-$DEFAULT_DATA_DIR}}"

# 确保数据目录存在
ensure_data_dir() {
    if [ ! -d "$DATA_DIR" ]; then
        echo "创建数据目录: $DATA_DIR"
        # 尝试普通创建，失败则用 sudo
        mkdir -p "$DATA_DIR" 2>/dev/null || sudo mkdir -p "$DATA_DIR"
        # 修改权限让当前用户可写
        sudo chown -R $(whoami):$(whoami) "$DATA_DIR" 2>/dev/null
    fi
}

case "$1" in
    start)
        ensure_data_dir
        echo "Milvus 数据目录: $DATA_DIR"
        echo "使用 Python: $PYTHON3"
        
        # 检查 milvus 模块是否存在
        if ! $PYTHON3 -c "from milvus import default_server" 2>/dev/null; then
            echo "✗ 错误: milvus 模块未安装"
            echo "请运行: pip3 install milvus"
            exit 1
        fi
        
        nohup $PYTHON3 -c "
import sys
sys.path.insert(0, '.')

from milvus import default_server

# 设置数据存储路径
default_server.set_base_dir('$DATA_DIR')

# 启动服务
default_server.start()
print('Milvus-Lite started on port 19530')
print('数据存储路径: $DATA_DIR')

import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    default_server.stop()
    print('Stopped')
" > /tmp/milvus.log 2>&1 &
        echo "Milvus-Lite 正在后台启动..."
        sleep 3
        
        # 检查是否启动成功
        if lsof -i :19530 > /dev/null 2>&1; then
            echo "✓ Milvus-Lite 启动成功"
            echo "日志文件: /tmp/milvus.log"
        else
            echo "✗ Milvus-Lite 启动失败，请查看日志:"
            cat /tmp/milvus.log
        fi
        ;;
    stop)
        echo "正在停止 Milvus-Lite..."
        $PYTHON3 -c "from milvus import default_server; default_server.stop(); print('Stopped')" 2>/dev/null
        
        # 强制杀掉占用端口的进程
        if command -v fuser > /dev/null 2>&1; then
            fuser -k 19530/tcp 2>/dev/null
        else
            # 备用方案：使用 lsof + kill
            PID=$(lsof -t -i :19530 2>/dev/null)
            if [ -n "$PID" ]; then
                kill -9 $PID 2>/dev/null
            fi
        fi
        
        echo "✓ Milvus-Lite 已停止"
        ;;
    status)
        echo "Milvus 数据目录: $DATA_DIR"
        if lsof -i :19530 > /dev/null 2>&1; then
            echo "✓ Milvus-Lite 正在运行"
            echo "进程信息:"
            lsof -i :19530
        else
            echo "✗ Milvus-Lite 未运行"
        fi
        ;;
    log)
        echo "查看 Milvus 日志:"
        cat /tmp/milvus.log 2>/dev/null || echo "日志文件不存在"
        ;;
    init)
        # 初始化 Collection（创建 memory_vectors collection）
        # Schema 定义：
        # - id (VarChar, PK)
        # - bot_id (VarChar)
        # - user_id (VarChar)
        # - memory_type (VarChar)
        # - created_at (Int64)
        # - content (VarChar)
        # - metadata (JSON) - 用于存储额外的元数据
        # - vector (FloatVector, dim=1536)
        
        COLLECTION_NAME="${2:-memory_vectors}"
        DIMENSION="${3:-1536}"
        MILVUS_HOST="${MILVUS_HOST:-localhost}"
        MILVUS_PORT="${MILVUS_PORT:-19530}"
        
        echo "正在创建 Collection: $COLLECTION_NAME (维度: $DIMENSION)"
        echo "Milvus 地址: $MILVUS_HOST:$MILVUS_PORT"
        
        $PYTHON3 << EOF
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 连接 Milvus
connections.connect("default", host="$MILVUS_HOST", port="$MILVUS_PORT")

collection_name = "$COLLECTION_NAME"
dimension = $DIMENSION

# 检查 collection 是否已存在
if utility.has_collection(collection_name):
    print(f"✗ Collection '{collection_name}' 已存在")
    print("如需重建，请先删除: ./scripts/milvus.sh drop $COLLECTION_NAME")
else:
    # 定义 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="bot_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]
    schema = CollectionSchema(fields, description="Memory vectors collection")
    
    # 创建 collection
    collection = Collection(name=collection_name, schema=schema)
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    
    print(f"✓ Collection '{collection_name}' 创建成功")
    print(f"  - 向量维度: {dimension}")
    print(f"  - 索引类型: IVF_FLAT")

connections.disconnect("default")
EOF
        ;;
    drop)
        # 删除 Collection
        COLLECTION_NAME="${2:-memory_vectors}"
        MILVUS_HOST="${MILVUS_HOST:-localhost}"
        MILVUS_PORT="${MILVUS_PORT:-19530}"
        
        echo "正在删除 Collection: $COLLECTION_NAME"
        
        $PYTHON3 << EOF
from pymilvus import connections, utility

connections.connect("default", host="$MILVUS_HOST", port="$MILVUS_PORT")

collection_name = "$COLLECTION_NAME"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✓ Collection '{collection_name}' 已删除")
else:
    print(f"✗ Collection '{collection_name}' 不存在")

connections.disconnect("default")
EOF
        ;;
    *)
        echo "Milvus-Lite 管理脚本"
        echo ""
        echo "用法: $0 {start|stop|status|log|init|drop} [参数]"
        echo ""
        echo "命令:"
        echo "  start   启动 Milvus-Lite 服务"
        echo "  stop    停止 Milvus-Lite 服务"
        echo "  status  查看服务状态"
        echo "  log     查看启动日志"
        echo "  init    创建 Collection (参数: [collection名称] [向量维度])"
        echo "  drop    删除 Collection (参数: [collection名称])"
        echo ""
        echo "示例:"
        echo "  $0 start                           # 使用默认路径启动"
        echo "  $0 start /path/to/data             # 指定数据目录启动"
        echo "  $0 init                            # 创建默认 memory_vectors collection (1536维)"
        echo "  $0 init my_collection 768          # 创建自定义 collection (768维)"
        echo "  $0 drop memory_vectors             # 删除 collection"
        echo "  MILVUS_DATA_DIR=/path/to/data $0 start  # 通过环境变量指定"
        echo ""
        echo "环境变量:"
        echo "  MILVUS_DATA_DIR  数据存储目录 (默认: $DEFAULT_DATA_DIR)"
        echo "  MILVUS_HOST      Milvus 服务地址 (默认: localhost)"
        echo "  MILVUS_PORT      Milvus 服务端口 (默认: 19530)"
        echo ""
        echo "默认数据目录: $DEFAULT_DATA_DIR"
        echo "当前数据目录: $DATA_DIR"
        echo "Python路径: $PYTHON3"
        exit 1
        ;;
esac