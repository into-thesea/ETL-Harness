# ============================================================================
# 启动 ETL-Harness 的沙箱层基础设施（OpenSandbox 服务端）
#
# 服务端以独立 venv 运行在本目录，不与项目 .venv 混装：
#   项目 .venv        只装轻量客户端 SDK（opensandbox）
#   本目录 .venv      装控制面（opensandbox-server，依赖 docker/fastapi/grpcio/...）
#
# 首次准备（若 .venv 不存在）：
#   python -m venv .venv
#   .venv\Scripts\python.exe -m pip install opensandbox-server==0.2.3
#   .venv\Scripts\opensandbox-server.exe init-config .\sandbox.toml --example docker
#
# 用法：
#   infra\opensandbox-server\start.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe  = Join-Path $here ".venv\Scripts\opensandbox-server.exe"
$conf = Join-Path $here "sandbox.toml"

if (-not (Test-Path $exe)) {
    Write-Host "未找到服务端：$exe" -ForegroundColor Red
    Write-Host "请先在本目录准备独立 venv：" -ForegroundColor Yellow
    Write-Host "  cd `"$here`""
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\python.exe -m pip install opensandbox-server==0.2.3"
    exit 1
}
if (-not (Test-Path $conf)) {
    Write-Host "未找到配置：$conf" -ForegroundColor Red
    exit 1
}

Write-Host "启动 OpenSandbox 服务端（Ctrl+C 停止）..." -ForegroundColor Cyan
& $exe --config $conf
