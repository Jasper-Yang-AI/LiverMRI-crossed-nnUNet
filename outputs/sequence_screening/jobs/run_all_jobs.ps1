$ErrorActionPreference = "Stop"
$JobsRoot = $PSScriptRoot
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS01\commands\run_RS01.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS02\commands\run_RS02.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS03\commands\run_RS03.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS04\commands\run_RS04.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS05\commands\run_RS05.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS06\commands\run_RS06.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS07\commands\run_RS07.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS08\commands\run_RS08.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS09\commands\run_RS09.ps1"))
& [System.IO.Path]::GetFullPath((Join-Path $JobsRoot "RS10\commands\run_RS10.ps1"))
