param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceRepoPath
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..\\..")
$targetRoot = Resolve-Path $SpaceRepoPath

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path $Source -Destination $Destination -Recurse -Force
}

Write-Host "Syncing Hugging Face Space files into $targetRoot"

Copy-Item (Join-Path $scriptRoot "app.py") (Join-Path $targetRoot "app.py") -Force
Copy-Item (Join-Path $scriptRoot "requirements.txt") (Join-Path $targetRoot "requirements.txt") -Force
Copy-Item (Join-Path $scriptRoot ".gitattributes") (Join-Path $targetRoot ".gitattributes") -Force

$assignmentTarget = Join-Path $targetRoot "assignments\\assignment-1"
New-Item -ItemType Directory -Force -Path $assignmentTarget | Out-Null

Copy-Tree `
    -Source (Join-Path $repoRoot "assignments\\assignment-1\\app") `
    -Destination $assignmentTarget

Copy-Tree `
    -Source (Join-Path $repoRoot "assignments\\assignment-1\\image\\artifacts") `
    -Destination (Join-Path $assignmentTarget "image")

Get-ChildItem `
    (Join-Path $assignmentTarget "image\\artifacts") `
    -Recurse `
    -Filter "*.png" `
    -File | Remove-Item -Force

Copy-Tree `
    -Source (Join-Path $repoRoot "assignments\\assignment-1\\image\\models") `
    -Destination (Join-Path $assignmentTarget "image")

Get-ChildItem $targetRoot -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Write-Host "Done."
Write-Host "Next steps inside the Space repo:"
Write-Host "  1. git lfs install"
Write-Host "  2. git add ."
Write-Host "  3. git commit -m 'Add Gradio image demo'"
Write-Host "  4. git push"
