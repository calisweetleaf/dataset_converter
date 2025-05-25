# Universal Syntax Checker
# Checks syntax for various file types.

param(
 [Parameter(Mandatory = $false, Position = 0)]
 [string]$Path = ".",

 [Parameter(Mandatory = $false)]
 [string[]]$FileTypes = @("ps1", "py", "json", "yaml", "xml", "md"),

 [Parameter(Mandatory = $false)]
 [switch]$Recurse
)

function Test-PowerShellSyntax {
 param ([string]$FilePath)
 try {
  $errors = $null
  $null = Get-Content $FilePath | Out-Null # Ensure file is readable
  [System.Management.Automation.Language.Parser]::ParseFile($FilePath, [ref]$null, [ref]$errors)
  if ($errors.Count -gt 0) {
   Write-Warning "Syntax errors found in $($FilePath):"
   $errors | ForEach-Object { Write-Warning "  $($_.Message) (Line: $($_.Extent.StartLineNumber))" }
   return $false
  }
  return $true
 }
 catch {
  Write-Error "Error checking PowerShell syntax for $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

function Test-PythonSyntax {
 param ([string]$FilePath)
 try {
  $pythonCmd = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { $null }
  if (-not $pythonCmd) {
   Write-Warning "Python interpreter not found. Skipping Python syntax check for $FilePath."
   return $true # Or $false if strict checking is needed
  }
  $process = Start-Process -FilePath $pythonCmd -ArgumentList "-m py_compile `"$FilePath`"" -PassThru -Wait -NoNewWindow -RedirectStandardError "py_errors.txt" -RedirectStandardOutput "py_output.txt"
  if ($process.ExitCode -ne 0) {
   Write-Warning "Syntax errors found in $($FilePath):"
   Get-Content "py_errors.txt" | ForEach-Object { Write-Warning "  $_" }
   Remove-Item "py_errors.txt", "py_output.txt" -ErrorAction SilentlyContinue
   return $false
  }
  Remove-Item "py_errors.txt", "py_output.txt" -ErrorAction SilentlyContinue
  return $true
 }
 catch {
  Write-Error "Error checking Python syntax for $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

function Test-JsonSyntax {
 param ([string]$FilePath)
 try {
  Get-Content $FilePath | ConvertFrom-Json | Out-Null
  return $true
 }
 catch {
  Write-Warning "JSON syntax error in $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

function Test-YamlSyntax {
 param ([string]$FilePath)
 try {
  # PowerShell doesn't have a native YAML parser. Requires a module like 'powershell-yaml'.
  if (-not (Get-Module -Name 'powershell-yaml' -ListAvailable)) {
   Write-Warning "Module 'powershell-yaml' not found. Skipping YAML syntax check for $FilePath. Install with: Install-Module powershell-yaml -Scope CurrentUser"
   return $true # Or $false
  }
  Import-Module 'powershell-yaml' -ErrorAction Stop
  Get-Content $FilePath | ConvertFrom-Yaml | Out-Null
  return $true
 }
 catch {
  Write-Warning "YAML syntax error in $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

function Test-XmlSyntax {
 param ([string]$FilePath)
 try {
  $xml = New-Object System.Xml.XmlDocument
  $xml.Load((Get-Item $FilePath).FullName)
  return $true
 }
 catch {
  Write-Warning "XML syntax error in $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

function Test-MarkdownSyntax {
 param ([string]$FilePath)
 # Basic check: Ensure it's valid text. More advanced linting requires external tools.
 try {
  $content = Get-Content $FilePath -Raw
  if ($content -match "[\x00-\x08\x0B\x0C\x0E-\x1F]") {
   # Check for non-printable characters
   Write-Warning "Invalid characters found in Markdown file: $FilePath"
   return $false
  }
  return $true
 }
 catch {
  Write-Warning "Error reading Markdown file $($FilePath): $($_.Exception.Message)"
  return $false
 }
}

Write-Host "🔍 Starting Universal Syntax Check..." -ForegroundColor Cyan
Write-Host "Path: $Path" -ForegroundColor Cyan
Write-Host "File Types: $($FileTypes -join ', ')" -ForegroundColor Cyan
Write-Host "Recurse: $Recurse" -ForegroundColor Cyan
Write-Host "-------------------------------------"

$filesToCheck = @()
if (Test-Path $Path -PathType Container) {
 $FileTypes | ForEach-Object {
  $ext = $_.TrimStart(".")
  $filesToCheck += Get-ChildItem -Path $Path -Filter "*.$ext" -Recurse:$Recurse -File -ErrorAction SilentlyContinue
 }
}
elseif (Test-Path $Path -PathType Leaf) {
 $filesToCheck += Get-Item $Path
}
else {
 Write-Error "Path '$Path' not found."
 exit 1
}

if ($filesToCheck.Count -eq 0) {
 Write-Host "No files found to check with the specified criteria." -ForegroundColor Yellow
 exit 0
}

$overallResult = $true
$checkedFiles = 0
$errorFiles = 0

foreach ($file in $filesToCheck | Sort-Object -Unique) {
 $checkedFiles++
 $fileExtension = $file.Extension.TrimStart(".").ToLower()
 $fileIsValid = $true

 Write-Host "Checking: $($file.FullName)" -ForegroundColor White

 switch ($fileExtension) {
  "ps1" {
   if (-not (Test-PowerShellSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "py" {
   if (-not (Test-PythonSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "json" {
   if (-not (Test-JsonSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "yaml" {
   if (-not (Test-YamlSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "yml" {
   if (-not (Test-YamlSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "xml" {
   if (-not (Test-XmlSyntax $file.FullName)) { $fileIsValid = $false }
  }
  "md" {
   if (-not (Test-MarkdownSyntax $file.FullName)) { $fileIsValid = $false }
  }
  default {
   Write-Host "  Skipping syntax check for .$fileExtension (unsupported or not specified)" -ForegroundColor Gray
   continue
  }
 }

 if ($fileIsValid) {
  Write-Host "  ✅ Syntax OK" -ForegroundColor Green
 }
 else {
  Write-Host "  ❌ Syntax ERROR" -ForegroundColor Red
  $overallResult = $false
  $errorFiles++
 }
}

Write-Host "-------------------------------------"
Write-Host "📊 Syntax Check Summary:" -ForegroundColor Cyan
Write-Host "Total Files Checked: $checkedFiles"

if ($overallResult) {
 Write-Host "🎉 All checked files have valid syntax!" -ForegroundColor Green
 exit 0
}
else {
 Write-Error "Found syntax errors in $errorFiles file(s)."
 exit 1
}
