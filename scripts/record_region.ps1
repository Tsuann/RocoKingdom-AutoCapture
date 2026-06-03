# 区域录制脚本 — 只录制屏幕指定区域
# 用法: .\scripts\record_region.ps1 -X 800 -Y 300 -W 640 -H 480 -Output videos/xiao_dujiaoshou.mp4
# 按 Ctrl+C 停止录制

param(
    [int]$X = 800,       # 区域左上角 X
    [int]$Y = 300,       # 区域左上角 Y
    [int]$W = 640,       # 宽度
    [int]$H = 480,       # 高度
    [string]$Output = "videos/recording.mp4",
    [int]$FPS = 30
)

# 检查 ffmpeg
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "需要安装 ffmpeg: winget install ffmpeg" -ForegroundColor Red
    Write-Host "或者用 OBS Studio (推荐，有图形界面)"
    exit 1
}

# 创建输出目录
$outDir = Split-Path $Output -Parent
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 屏幕区域录制" -ForegroundColor Cyan
Write-Host " 区域: ${X},${Y} ${W}x${H}" -ForegroundColor Yellow
Write-Host " 输出: $Output" -ForegroundColor Yellow
Write-Host " 按 Ctrl+C 停止录制" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# FFmpeg gdigrab 区域录制
$ffmpegArgs = @(
    "-f", "gdigrab",
    "-framerate", $FPS,
    "-offset_x", $X,
    "-offset_y", $Y,
    "-video_size", "${W}x${H}",
    "-i", "desktop",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-y",
    $Output
)

Write-Host "`n开始录制..." -ForegroundColor Green
ffmpeg @ffmpegArgs
Write-Host "录制完成: $Output" -ForegroundColor Green
