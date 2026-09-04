$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "$env:LOCALAPPDATA\Whisper\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  $py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $py)) {
  throw "Не найден Python venv. Сначала запустите setup.bat"
}

& $py -m pip install --upgrade pyinstaller
& $py -m PyInstaller --noconfirm --clean WhisperTranscribe.spec

$dist = Join-Path $PSScriptRoot "dist\WhisperTranscribe"
if (-not (Test-Path (Join-Path $dist "WhisperTranscribe.exe"))) {
  throw "Сборка не создала WhisperTranscribe.exe"
}

$ffmpegInternal = Join-Path $dist "_internal\ffmpeg.exe"
if ((Test-Path $ffmpegInternal) -and -not (Test-Path (Join-Path $dist "ffmpeg.exe"))) {
  Copy-Item $ffmpegInternal (Join-Path $dist "ffmpeg.exe")
}

$zip = Join-Path $PSScriptRoot "dist\WhisperTranscribe-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $dist "*") -DestinationPath $zip -CompressionLevel Optimal
Write-Host "ZIP: $zip"
Get-Item $zip | Select-Object FullName, Length
