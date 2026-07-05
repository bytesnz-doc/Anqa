$ErrorActionPreference = "Stop"

py -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name "anqa-annotator" `
  --paths "src" `
  "run_anqa_desktop.py"

Write-Host "Built dist\\anqa-annotator.exe"
