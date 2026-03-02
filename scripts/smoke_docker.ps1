param(
  [int]$HostPort = 8003,
  [string]$ImageTag = "omnibench_env:v0.1.1",
  [string]$ContainerName = "omnibench_env",
  [string]$BaseUrl = "",
  [string]$DockerfilePath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  $BaseUrl = "http://127.0.0.1:$HostPort"
}

# Elegir Dockerfile (prefiere server/Dockerfile si existe)
if ([string]::IsNullOrWhiteSpace($DockerfilePath)) {
  if (Test-Path ".\server\Dockerfile") { $DockerfilePath = ".\server\Dockerfile" }
  elseif (Test-Path ".\Dockerfile") { $DockerfilePath = ".\Dockerfile" }
  else { throw "No Dockerfile found (expected .\server\Dockerfile or .\Dockerfile)." }
}

Write-Host "==> Building image: $ImageTag (Dockerfile=$DockerfilePath)"
docker build -t $ImageTag -f $DockerfilePath .

Write-Host "==> Removing old container (if exists): $ContainerName"
docker rm -f $ContainerName 2>$null | Out-Null

Write-Host "==> Running container: $ContainerName (host $HostPort -> container 8000)"
docker run -d --name $ContainerName -p "$HostPort`:8000" $ImageTag | Out-Null

Write-Host "==> Waiting for server..."
Start-Sleep -Seconds 2

Write-Host "==> Smoke 7/7 via uv (BaseUrl=$BaseUrl)"
uv run --project . python scripts/smoke_test_all_domains.py --base-url $BaseUrl --verbose

Write-Host "==> PASS: docker + smoke"
