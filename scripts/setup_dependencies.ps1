# Windows equivalent of setup_dependencies.sh — see that file for rationale.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$ThirdPartyDir = Join-Path $RootDir "third_party"
New-Item -ItemType Directory -Force -Path $ThirdPartyDir | Out-Null

function Clone-Pinned {
    param([string]$Name, [string]$Url, [string]$Commit)

    $Dest = Join-Path $ThirdPartyDir $Name
    if (Test-Path (Join-Path $Dest ".git")) {
        Write-Host "[$Name] already present at $Dest, skipping clone (delete it to re-fetch)"
        return
    }

    Write-Host "[$Name] cloning $Url @ $Commit"
    git clone $Url $Dest
    git -C $Dest checkout $Commit
}

Clone-Pinned -Name "ConVA" -Url "https://github.com/hr-jin/ConVA.git" -Commit "9484868882cd42752d1b9f27edcb1d9a1d41eb99"
Clone-Pinned -Name "representation-engineering" -Url "https://github.com/andyzoujm/representation-engineering.git" -Commit "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e"
# ModPlural benchmark data (input/*.json) that pipeline/run_*.py --input points
# to. Vanilla/MoE/ModPlural/Ethos baseline numbers are cited directly from
# their papers, not reproduced, so nothing here imports this repo's code.
Clone-Pinned -Name "modular_pluralism" -Url "https://github.com/BunsenFeng/modular_pluralism.git" -Commit "56bd05d2e6d824e93a5c0bee5ac26b56aeb299aa"

Write-Host ""
Write-Host "Done. third_party/ now contains:"
Get-ChildItem $ThirdPartyDir | Select-Object -ExpandProperty Name
Write-Host ""
Write-Host "representation-engineering is pip-installable (MIT licensed): consider"
Write-Host "    pip install -e third_party/representation-engineering"
Write-Host "ConVA and modular_pluralism ship no LICENSE file (all-rights-reserved by"
Write-Host "default upstream) and are NOT pip packages -- this repo only imports from"
Write-Host "them at runtime via sys.path, never copies or redistributes their code."
