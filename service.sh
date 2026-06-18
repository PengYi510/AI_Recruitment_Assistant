#!/bin/bash
#
# HR Agent 服务一键管理脚本
#
# 用法:
#   bash service.sh start    # 启动前后端服务（后台运行）
#   bash service.sh stop     # 停止所有服务
#   bash service.sh restart  # 重启所有服务
#   bash service.sh status   # 查看服务状态
#   bash service.sh logs     # 查看实时日志（Ctrl+C 退出）
#
# 服务说明:
#   - 后端 API (http_server.py):  FastAPI + Uvicorn, 端口 8003
#   - 前端 Web (frontend_server.py): Flask, 端口 9033
#
# 启动后访问: http://localhost:9033
#

set -e

# ── 配置 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"
ADMIN_PID_FILE="$SCRIPT_DIR/.admin.pid"
LOG_DIR="$SCRIPT_DIR/logs"

BACKEND_PORT=${BACKEND_PORT:-8003}
FRONTEND_PORT=${FRONTEND_PORT:-9033}
ADMIN_PORT_START=${ADMIN_PORT_START:-9035}

# 自动检测 Python（优先用虚拟环境的）
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [ -f "$SCRIPT_DIR/../venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../venv/bin/python"
else
    PYTHON="python"
fi

# ── 工具函数 ──────────────────────────────────────────────────────────────────

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

is_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # PID 文件存在但进程已死，清理
        rm -f "$pid_file"
    fi
    return 1
}

get_pid() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

# 检测端口是否已被监听（优先 lsof，回退 nc / /dev/tcp）
port_listening() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti ":$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0 || return 1
    elif command -v nc >/dev/null 2>&1; then
        nc -z 127.0.0.1 "$port" >/dev/null 2>&1 && return 0 || return 1
    else
        (echo > "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1 && return 0 || return 1
    fi
}

# 轮询等待端口就绪，期间每 2s 原地刷新进度，避免让人误以为卡死
# 参数: $1=端口 $2=进程PID $3=最长超时秒 $4=服务名
# 返回: 0=ready 1=crashed 2=timeout
wait_port_ready() {
    local port=$1 pid=$2 timeout=$3 name=$4
    local start elapsed hint
    start=$(date +%s)
    local hint_default="正在启动并等待端口 $port 就绪"
    local hint_backend="正在加载模型(BGE-M3 2.2GB + BLIP)并等待端口 $port 就绪"
    while :; do
        # 进程已退出 -> 崩溃
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            printf '\n'
            return 1
        fi
        if port_listening "$port"; then
            printf '\n'
            return 0
        fi
        elapsed=$(( $(date +%s) - start ))
        if [ "$elapsed" -ge "$timeout" ]; then
            printf '\n'
            return 2
        fi
        if [ "$name" = "后端" ]; then hint="$hint_backend"; else hint="$hint_default"; fi
        # \r 回到行首原地刷新，不刷屏
        printf '\r  [%s] %s ... 已等待 %3ss / 最长 %ss' "$name" "$hint" "$elapsed" "$timeout"
        sleep 2
    done
}

# ── start ─────────────────────────────────────────────────────────────────────

do_start() {
    ensure_log_dir
    echo "========================================"
    echo "  HR Agent 服务启动"
    echo "========================================"

    # 启动后端
    if is_running "$BACKEND_PID_FILE"; then
        echo "[后端] 已在运行 (PID: $(get_pid "$BACKEND_PID_FILE"))，跳过"
    else
        echo "[后端] 启动 http_server.py (端口 $BACKEND_PORT) ..."
        nohup $PYTHON http_server.py > "$LOG_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo $BACKEND_PID > "$BACKEND_PID_FILE"
        echo "[后端] 进程已拉起 (PID: $BACKEND_PID)，等待端口 $BACKEND_PORT 就绪中（下方每 2s 刷新进度）..."
        # 后端首启需加载 BGE-M3(2.2GB)+BLIP，CPU 上较慢，给 180s 超时
        wait_port_ready "$BACKEND_PORT" "$BACKEND_PID" 180 "后端"
        case $? in
            0) echo "[后端] ✅ 启动成功 (PID: $BACKEND_PID, 端口: $BACKEND_PORT)" ;;
            1) echo "[后端] ❌ 进程已退出，启动失败。错误日志末尾："
               tail -n 8 "$LOG_DIR/backend.log" 2>/dev/null | sed 's/^/    /'
               rm -f "$BACKEND_PID_FILE" ;;
            2) echo "[后端] ⚠ 等待超时，端口仍未就绪（进程可能仍在加载模型）。日志末尾："
               tail -n 5 "$LOG_DIR/backend.log" 2>/dev/null | sed 's/^/    /'
               echo "[后端]   提示：稍后用 bash service.sh status 复查。" ;;
        esac
    fi

    # 启动前端
    if is_running "$FRONTEND_PID_FILE"; then
        echo "[前端] 已在运行 (PID: $(get_pid "$FRONTEND_PID_FILE"))，跳过"
    else
        echo "[前端] 启动 frontend_server.py (端口 $FRONTEND_PORT) ..."
        nohup $PYTHON frontend_server.py > "$LOG_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > "$FRONTEND_PID_FILE"
        echo "[前端] 进程已拉起 (PID: $FRONTEND_PID)，等待端口 $FRONTEND_PORT 就绪中..."
        wait_port_ready "$FRONTEND_PORT" "$FRONTEND_PID" 30 "前端"
        case $? in
            0) echo "[前端] ✅ 启动成功 (PID: $FRONTEND_PID, 端口: $FRONTEND_PORT)" ;;
            1) echo "[前端] ❌ 进程已退出，启动失败。错误日志末尾："
               tail -n 8 "$LOG_DIR/frontend.log" 2>/dev/null | sed 's/^/    /'
               rm -f "$FRONTEND_PID_FILE" ;;
            2) echo "[前端] ⚠ 等待超时，端口仍未就绪。请检查 $LOG_DIR/frontend.log" ;;
        esac
    fi

    # 启动管理面板
    if is_running "$ADMIN_PID_FILE"; then
        echo "[管理面板] 已在运行 (PID: $(get_pid "$ADMIN_PID_FILE"))，跳过"
    else
        echo "[管理面板] 启动 admin_server.py (端口 $ADMIN_PORT_START+) ..."
        nohup $PYTHON admin_server.py > "$LOG_DIR/admin.log" 2>&1 &
        ADMIN_PID=$!
        echo $ADMIN_PID > "$ADMIN_PID_FILE"
        echo "[管理面板] 进程已拉起 (PID: $ADMIN_PID)，等待端口 $ADMIN_PORT_START 就绪中..."
        wait_port_ready "$ADMIN_PORT_START" "$ADMIN_PID" 30 "管理面板"
        case $? in
            0) # 从日志中提取实际使用的端口
               ACTUAL_ADMIN_PORT=$(grep -oP '(?<=port )\d+' "$LOG_DIR/admin.log" 2>/dev/null | tail -1)
               [ -z "$ACTUAL_ADMIN_PORT" ] && ACTUAL_ADMIN_PORT=$ADMIN_PORT_START
               echo "[管理面板] ✅ 启动成功 (PID: $ADMIN_PID, 端口: $ACTUAL_ADMIN_PORT)" ;;
            1) echo "[管理面板] ❌ 进程已退出，启动失败。错误日志末尾："
               tail -n 8 "$LOG_DIR/admin.log" 2>/dev/null | sed 's/^/    /'
               rm -f "$ADMIN_PID_FILE"
               ACTUAL_ADMIN_PORT="(启动失败)" ;;
            2) echo "[管理面板] ⚠ 等待超时，端口仍未就绪。请检查 $LOG_DIR/admin.log"
               ACTUAL_ADMIN_PORT="(待确认)" ;;
        esac
    fi

    # 确保 resume_data 目录存在
    mkdir -p "$SCRIPT_DIR/resume_data"

    echo ""
    echo "========================================"
    echo "  ✅ 服务已启动"
    echo "  前端页面:   http://localhost:$FRONTEND_PORT"
    echo "  后端API:    http://localhost:$BACKEND_PORT"
    echo "  管理面板:   http://localhost:${ACTUAL_ADMIN_PORT:-$ADMIN_PORT_START}"
    echo "  简历目录:   $SCRIPT_DIR/resume_data/"
    echo "  日志目录:   $LOG_DIR/"
    echo "  管理员账号: admin / admin"
    echo "========================================"
}

# ── stop ──────────────────────────────────────────────────────────────────────

do_stop() {
    echo "========================================"
    echo "  HR Agent 服务停止"
    echo "========================================"

    # 停止管理面板
    if is_running "$ADMIN_PID_FILE"; then
        local pid
        pid=$(get_pid "$ADMIN_PID_FILE")
        echo "[管理面板] 停止进程 (PID: $pid) ..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$ADMIN_PID_FILE"
        echo "[管理面板] 已停止"
    else
        echo "[管理面板] 未在运行"
    fi

    # 停止前端
    if is_running "$FRONTEND_PID_FILE"; then
        local pid
        pid=$(get_pid "$FRONTEND_PID_FILE")
        echo "[前端] 停止进程 (PID: $pid) ..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$FRONTEND_PID_FILE"
        echo "[前端] 已停止"
    else
        echo "[前端] 未在运行"
    fi

    # 停止后端
    if is_running "$BACKEND_PID_FILE"; then
        local pid
        pid=$(get_pid "$BACKEND_PID_FILE")
        echo "[后端] 停止进程 (PID: $pid) ..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$BACKEND_PID_FILE"
        echo "[后端] 已停止"
    else
        echo "[后端] 未在运行"
    fi

    # 兜底：按端口清理残余进程
    local remaining
    remaining=$(lsof -ti :$BACKEND_PORT 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        echo "[清理] 杀死占用端口 $BACKEND_PORT 的进程: $remaining"
        echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi
    remaining=$(lsof -ti :$FRONTEND_PORT 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        echo "[清理] 杀死占用端口 $FRONTEND_PORT 的进程: $remaining"
        echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi

    echo ""
    echo "  ✅ 所有服务已停止"
    echo "========================================"
}

# ── restart ───────────────────────────────────────────────────────────────────

do_restart() {
    do_stop
    sleep 1
    do_start
}

# ── status ────────────────────────────────────────────────────────────────────

do_status() {
    echo "========================================"
    echo "  HR Agent 服务状态"
    echo "========================================"

    if is_running "$BACKEND_PID_FILE"; then
        echo "  [后端]     ✅ 运行中 (PID: $(get_pid "$BACKEND_PID_FILE"), 端口: $BACKEND_PORT)"
    else
        echo "  [后端]     ❌ 未运行"
    fi

    if is_running "$FRONTEND_PID_FILE"; then
        echo "  [前端]     ✅ 运行中 (PID: $(get_pid "$FRONTEND_PID_FILE"), 端口: $FRONTEND_PORT)"
    else
        echo "  [前端]     ❌ 未运行"
    fi

    if is_running "$ADMIN_PID_FILE"; then
        echo "  [管理面板] ✅ 运行中 (PID: $(get_pid "$ADMIN_PID_FILE"), 端口: $ADMIN_PORT_START+)"
    else
        echo "  [管理面板] ❌ 未运行"
    fi

    echo "========================================"
}

# ── logs ──────────────────────────────────────────────────────────────────────

do_logs() {
    ensure_log_dir
    echo "实时日志（Ctrl+C 退出）:"
    echo "========================================"
    tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" "$LOG_DIR/admin.log" 2>/dev/null || \
        echo "日志文件不存在，请先启动服务"
}

# ── 主入口 ────────────────────────────────────────────────────────────────────

case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    logs)
        do_logs
        ;;
    *)
        echo "用法: bash $0 {start|stop|restart|status|logs}"
        echo ""
        echo "  start   - 后台启动前后端服务"
        echo "  stop    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  status  - 查看运行状态"
        echo "  logs    - 查看实时日志"
        echo ""
        echo "启动后访问: http://localhost:9033"
        exit 1
        ;;
esac
