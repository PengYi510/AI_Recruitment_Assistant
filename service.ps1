<#
.SYNOPSIS
    HR Agent 服务一键管理脚本（Windows PowerShell 版，对齐 service.sh）

.DESCRIPTION
    用法:
      .\service.ps1 start      # 启动前端、后端、管理面板（后台运行）
      .\service.ps1 stop       # 停止所有服务
      .\service.ps1 restart    # 重启所有服务
      .\service.ps1 status     # 查看服务状态
      .\service.ps1 logs       # 查看实时日志（Ctrl+C 退出）

    服务说明:
      - 后端 API (http_server.py):      FastAPI + Uvicorn, 端口 8003
      - 前端 Web (frontend_server.py):  Flask,            端口 9033
      - 管理面板 (admin_server.py):     FastAPI + Uvicorn, 端口 9035 起（自动递增）

    启动后访问: http://localhost:9033
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs')]
    [string]$Action = 'help'
)

$ErrorActionPreference = 'Stop'

# ── 配置 ──────────────────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$BackendPidFile  = Join-Path $ScriptDir '.backend.pid'
$FrontendPidFile = Join-Path $ScriptDir '.frontend.pid'
$AdminPidFile    = Join-Path $ScriptDir '.admin.pid'
$LogDir          = Join-Path $ScriptDir 'logs'

$BackendPort  = if ($env:BACKEND_PORT)  { [int]$env:BACKEND_PORT }  else { 8003 }
$FrontendPort = if ($env:FRONTEND_PORT) { [int]$env:FRONTEND_PORT } else { 9033 }
$AdminPortStart = if ($env:ADMIN_PORT_START) { [int]$env:ADMIN_PORT_START } else { 9035 }

# 自动检测 Python（可用 $env:HR_PYTHON 显式指定）
# 注意：需要选择已安装项目依赖(fastapi/torch/transformers...)的解释器。
# 本机已确认 C:\Python314 装齐依赖；venv313 为空环境，故默认优先 Python314。
function Resolve-Python {
    if ($env:HR_PYTHON -and (Test-Path $env:HR_PYTHON)) { return $env:HR_PYTHON }
    if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'))) {
        return (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe')
    }
    foreach ($candidate in @(
        'C:\Python314\python.exe',
        (Join-Path $ScriptDir 'venv313\Scripts\python.exe'),
        (Join-Path $ScriptDir 'venv\Scripts\python.exe')
    )) {
        if (Test-Path $candidate) { return $candidate }
    }
    return 'python'
}
$Python = Resolve-Python

# ── 工具函数 ──────────────────────────────────────────────────────────────────
function Ensure-LogDir { if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null } }

function Get-PidFromFile([string]$PidFile) {
    if (Test-Path $PidFile) { return (Get-Content $PidFile -Raw).Trim() }
    return $null
}

function Test-ServiceRunning([string]$PidFile) {
    $pidVal = Get-PidFromFile $PidFile
    if ($pidVal) {
        $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
        if ($proc) { return $true }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
    return $false
}

function Test-PortListening([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$c
}

# 轮询等待端口就绪；进程中途退出则提前判失败
# 等待期间每隔 ProgressEvery 秒打印一次实时进度，避免让人误以为卡死
function Wait-PortReady([int]$Port, [int]$PidVal, [int]$TimeoutSec, [string]$Name = '服务', [int]$ProgressEvery = 2) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $start = Get-Date
    $lastTick = -1
    while ((Get-Date) -lt $deadline) {
        if ($PidVal -and -not (Get-Process -Id $PidVal -ErrorAction SilentlyContinue)) {
            Write-Host ""   # 进度行后换行，避免和后续输出黏连
            return 'crashed'   # 进程已退出
        }
        if (Test-PortListening $Port) {
            Write-Host ""   # 进度行后换行
            return 'ready'
        }

        # 每 ProgressEvery 秒打印一次进度（同一行原地刷新）
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        $tick = [int]($elapsed / $ProgressEvery)
        if ($tick -ne $lastTick) {
            $lastTick = $tick
            $hint = "正在启动并等待端口 $Port 就绪"
            if ($Name -eq '后端') { $hint = "正在加载模型(BGE-M3 2.2GB + BLIP)并等待端口 $Port 就绪" }
            # \r 回到行首，原地刷新进度，不刷屏
            Write-Host ("`r  [{0}] {1} ... 已等待 {2,3}s / 最长 {3}s" -f $Name, $hint, $elapsed, $TimeoutSec) -NoNewline -ForegroundColor DarkGray
        }
        Start-Sleep -Milliseconds 800
    }
    Write-Host ""   # 进度行后换行
    return 'timeout'
}

function Stop-ByPidFile([string]$Name, [string]$PidFile) {
    if (Test-ServiceRunning $PidFile) {
        $pidVal = Get-PidFromFile $PidFile
        Write-Host "[$Name] 停止进程 (PID: $pidVal) ..."
        Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "[$Name] 已停止"
    }
    else {
        Write-Host "[$Name] 未在运行"
    }
}

function Kill-Port([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            Write-Host "[清理] 杀死占用端口 $Port 的进程 (PID: $_)"
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
    }
}

function Start-Service-Bg([string]$Name, [string]$Script, [string]$LogName, [string]$PidFile, [int]$Port, [int]$TimeoutSec) {
    # 端口已被占用 -> 视为已运行（即使 pid 文件丢失也不重复拉起）
    if (Test-PortListening $Port) {
        Write-Host "[$Name] 端口 $Port 已在监听，跳过（疑似已运行）"
        return
    }
    if (Test-ServiceRunning $PidFile) {
        Write-Host "[$Name] 已在运行 (PID: $(Get-PidFromFile $PidFile))，跳过"
        return
    }
    $logPath = Join-Path $LogDir $LogName
    Write-Host "[$Name] 启动 $Script ...（首启需加载模型，最长等待 ${TimeoutSec}s）"
    $proc = Start-Process -FilePath $Python -ArgumentList $Script `
        -WorkingDirectory $ScriptDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err"
    $pidVal = $proc.Id
    $pidVal | Out-File -FilePath $PidFile -Encoding ascii -NoNewline
    Write-Host "[$Name] 进程已拉起 (PID: $pidVal)，等待端口 $Port 就绪中（下方每 2s 刷新进度）..." -ForegroundColor DarkGray

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $result = Wait-PortReady -Port $Port -PidVal $pidVal -TimeoutSec $TimeoutSec -Name $Name
    $sw.Stop()
    $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)

    switch ($result) {
        'ready'   { Write-Host "[$Name] 启动成功 (PID: $pidVal, 端口: $Port, 耗时 ${secs}s)" }
        'crashed' {
            Write-Host "[$Name] X 进程已退出，启动失败。错误日志末尾："
            if (Test-Path "$logPath.err") { Get-Content "$logPath.err" -Tail 8 | ForEach-Object { Write-Host "    $_" } }
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        }
        'timeout' {
            Write-Host "[$Name] ! 等待 ${secs}s 端口 $Port 仍未就绪（进程仍在运行，可能模型仍在加载）。日志末尾："
            if (Test-Path $logPath) { Get-Content $logPath -Tail 5 | ForEach-Object { Write-Host "    $_" } }
            Write-Host "[$Name]   提示：稍后用 .\service.ps1 status 复查；首次加载 BGE-M3/BLIP 可能较慢。"
        }
    }
}

# ── start ─────────────────────────────────────────────────────────────────────
function Do-Start {
    Ensure-LogDir
    Write-Host '========================================'
    Write-Host '  HR Agent 服务启动'
    Write-Host '========================================'
    Write-Host "Python: $Python"

    # 后端首启需加载 BGE-M3(2.2GB)+BLIP，CPU 上较慢，给 180s 超时
    Start-Service-Bg '后端'     'http_server.py'     'backend.log'  $BackendPidFile  $BackendPort  180
    Start-Service-Bg '前端'     'frontend_server.py' 'frontend.log' $FrontendPidFile $FrontendPort 30
    Start-Service-Bg '管理面板' 'admin_server.py'    'admin.log'    $AdminPidFile    $AdminPortStart 30

    if (-not (Test-Path (Join-Path $ScriptDir 'resume_data'))) {
        New-Item -ItemType Directory -Path (Join-Path $ScriptDir 'resume_data') | Out-Null
    }

    Write-Host ''
    Write-Host '========================================'
    Write-Host '  服务已启动'
    Write-Host "  前端页面:   http://localhost:$FrontendPort"
    Write-Host "  后端API:    http://localhost:$BackendPort"
    Write-Host "  管理面板:   http://localhost:$AdminPortStart  (实际端口见 logs\admin.log)"
    Write-Host "  日志目录:   $LogDir"
    Write-Host '  管理员账号: admin / admin'
    Write-Host '========================================'
}

# ── stop ──────────────────────────────────────────────────────────────────────
function Do-Stop {
    Write-Host '========================================'
    Write-Host '  HR Agent 服务停止'
    Write-Host '========================================'
    Stop-ByPidFile '管理面板' $AdminPidFile
    Stop-ByPidFile '前端'     $FrontendPidFile
    Stop-ByPidFile '后端'     $BackendPidFile

    # 兜底：按端口清理残余
    Kill-Port $BackendPort
    Kill-Port $FrontendPort
    foreach ($p in $AdminPortStart..($AdminPortStart + 5)) { Kill-Port $p }

    Write-Host ''
    Write-Host '  所有服务已停止'
    Write-Host '========================================'
}

# ── restart ───────────────────────────────────────────────────────────────────
function Do-Restart { Do-Stop; Start-Sleep -Seconds 2; Do-Start }

# ── status ────────────────────────────────────────────────────────────────────
function Do-Status {
    Write-Host '========================================'
    Write-Host '  HR Agent 服务状态'
    Write-Host '========================================'
    if (Test-ServiceRunning $BackendPidFile)  { Write-Host "  [后端]     运行中 (PID: $(Get-PidFromFile $BackendPidFile), 端口: $BackendPort)" }  else { Write-Host '  [后端]     未运行' }
    if (Test-ServiceRunning $FrontendPidFile) { Write-Host "  [前端]     运行中 (PID: $(Get-PidFromFile $FrontendPidFile), 端口: $FrontendPort)" } else { Write-Host '  [前端]     未运行' }
    if (Test-ServiceRunning $AdminPidFile)    { Write-Host "  [管理面板] 运行中 (PID: $(Get-PidFromFile $AdminPidFile), 端口: $AdminPortStart+)" }  else { Write-Host '  [管理面板] 未运行' }
    Write-Host '========================================'
}

# ── logs ──────────────────────────────────────────────────────────────────────
function Do-Logs {
    Ensure-LogDir
    Write-Host '实时日志（Ctrl+C 退出）:'
    Write-Host '========================================'
    Get-Content (Join-Path $LogDir 'backend.log'), (Join-Path $LogDir 'frontend.log'), (Join-Path $LogDir 'admin.log') -Tail 20 -Wait -ErrorAction SilentlyContinue
}

# ── 主入口 ────────────────────────────────────────────────────────────────────
switch ($Action) {
    'start'   { Do-Start }
    'stop'    { Do-Stop }
    'restart' { Do-Restart }
    'status'  { Do-Status }
    'logs'    { Do-Logs }
    default {
        Write-Host "用法: .\service.ps1 {start|stop|restart|status|logs}"
        Write-Host ''
        Write-Host '  start   - 后台启动前端、后端、管理面板'
        Write-Host '  stop    - 停止所有服务'
        Write-Host '  restart - 重启所有服务'
        Write-Host '  status  - 查看运行状态'
        Write-Host '  logs    - 查看实时日志'
        Write-Host ''
        Write-Host '启动后访问: http://localhost:9033'
    }
}
