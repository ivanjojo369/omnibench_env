param(
  [string]$BaseUrl = "http://127.0.0.1:8003",
  [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$cmd = @(
  "run", "--project", ".",
  "python", "scripts/smoke_test_all_domains.py",
  "--base-url", $BaseUrl
)

if ($Verbose) { $cmd += "--verbose" }

uv @cmd
