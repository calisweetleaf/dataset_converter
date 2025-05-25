# File Finalizer Check
# Performs final checks on files before distribution or commit.

param(
 [Parameter(Mandatory = $false, Position = 0)]
 [string]$Path = ".",

 [Parameter(Mandatory = $false)]
 [string[]]$FileTypes = @("ps1", "py", "sh", "md", "txt", "json", "yaml", "xml"),

 [Parameter(Mandatory = $false)]
 [switch]$Recurse,

 [Parameter(Mandatory = $false)]
 [string[]]$ForbiddenKeywords = @("TODO:", "FIXME:", "TEMP:", "DEBUG:", "HACK:", "REMOVEBEFORECOMMIT", "DONOTDISTRIBUTE", "PASSWORD", "SECRET_KEY"),

 [Parameter(Mandatory = $false)]
 [long]$MaxSizeKB = 1024, # Maximum file size in KB (1MB)

 [Parameter(Mandatory = $false)]
 [switch]$CheckTrailingWhitespace,

 [Parameter(Mandatory = $false)]
 [switch]$CheckUtf8Bom, # Check for UTF-8 BOM (Byte Order Mark)
    
 [Parameter(Mandatory = $false)]
 [switch]$EnsureFinalNewline,

 [Parameter(Mandatory = $false)]
 [string]$ReportFile = ""
)

$startTime = Get-Date
Write-Host "🧐 Starting File Finalizer Check..." -ForegroundColor Cyan
Write-Host "Timestamp: $startTime" -ForegroundColor Cyan
Write-Host "Path: $Path" -ForegroundColor Cyan
Write-Host "File Types: $($FileTypes -join ', ')" -ForegroundColor Cyan
Write-Host "Recurse: $Recurse" -ForegroundColor Cyan
Write-Host "Forbidden Keywords: $($ForbiddenKeywords -join ', ')" -ForegroundColor Cyan
Write-Host "Max Size (KB): $MaxSizeKB" -ForegroundColor Cyan
Write-Host "Check Trailing Whitespace: $CheckTrailingWhitespace" -ForegroundColor Cyan
Write-Host "Check UTF-8 BOM: $CheckUtf8Bom" -ForegroundColor Cyan
Write-Host "Ensure Final Newline: $EnsureFinalNewline" -ForegroundColor Cyan
Write-Host "-------------------------------------"

$filesToFinalize = @()
if (Test-Path $Path -PathType Container) {
 $FileTypes | ForEach-Object {
  $ext = $_.TrimStart(".")
  $filesToFinalize += Get-ChildItem -Path $Path -Filter "*.$ext" -Recurse:$Recurse -File -ErrorAction SilentlyContinue
 }
}
elseif (Test-Path $Path -PathType Leaf) {
 $filesToFinalize += Get-Item $Path
}
else {
 Write-Error "Path '$Path' not found."
 exit 1
}

if ($filesToFinalize.Count -eq 0) {
 Write-Host "No files found to finalize with the specified criteria." -ForegroundColor Yellow
 exit 0
}

$reportLines = @()
$overallIssues = 0
$filesWithIssues = 0

foreach ($file in $filesToFinalize | Sort-Object -Unique) {
 Write-Host "Finalizing: $($file.FullName)" -ForegroundColor White
 $issuesInFile = 0
 $fileReport = @()

 # 1. Check File Size
 if (($file.Length / 1KB) -gt $MaxSizeKB) {
  $msg = "  ⚠️ File size ($($file.Length / 1KB) KB) exceeds maximum ($MaxSizeKB KB)."
  Write-Warning $msg
  $fileReport += $msg
  $issuesInFile++
 }

 # Read content for other checks
 $fileContentBytes = Get-Content $file.FullName -AsByteStream -Raw -ErrorAction SilentlyContinue
 if (-not $fileContentBytes) {
  $msg = "  Could not read file or file is empty: $($file.FullName)"
  Write-Warning $msg
  $fileReport += $msg
  # Allow empty files, but skip content-based checks
 }
 else {
  $lines = Get-Content $file.FullName -Encoding Default

  # 2. Check for Forbidden Keywords (with line numbers)
  foreach ($keyword in $ForbiddenKeywords) {
   for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match $keyword) {
     $msg = "  ⚠️ Found forbidden keyword '$keyword' on line $($i+1)."
     Write-Warning $msg
     $fileReport += $msg
     $issuesInFile++
    }
   }
  }

  # 3. Check for Trailing Whitespace
  if ($CheckTrailingWhitespace) {
   for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "\s+$") {
     $msg = "  ⚠️ Trailing whitespace found on line $($i+1)."
     Write-Warning $msg
     $fileReport += $msg
     $issuesInFile++
     break
    }
   }
  }

  # 4. Check for UTF-8 BOM
  if ($CheckUtf8Bom) {
   if ($fileContentBytes.Count -ge 3 -and $fileContentBytes[0] -eq 0xEF -and $fileContentBytes[1] -eq 0xBB -and $fileContentBytes[2] -eq 0xBF) {
    $msg = "  ⚠️ File contains UTF-8 BOM."
    Write-Warning $msg
    $fileReport += $msg
    $issuesInFile++
   }
  }

  # 5. Ensure Final Newline
  if ($EnsureFinalNewline) {
   if ($fileContentBytes.Count -gt 0 -and $fileContentBytes[-1] -ne 0x0A) {
    if ($fileContentBytes.Count -lt 2 -or ($fileContentBytes[-2] -ne 0x0D -or $fileContentBytes[-1] -ne 0x0A)) {
     $msg = "  ⚠️ File does not end with a newline character."
     Write-Warning $msg
     $fileReport += $msg
     $issuesInFile++
    }
   }
  }

  # 6. Check for non-UTF8 encoding (simple heuristic)
  try {
   $null = [System.Text.Encoding]::UTF8.GetString($fileContentBytes)
  }
  catch {
   $msg = "  ⚠️ File may not be valid UTF-8 encoding."
   Write-Warning $msg
   $fileReport += $msg
   $issuesInFile++
  }

  # 7. Check file permissions (read-only, executable)
  if ($file.Attributes -band [System.IO.FileAttributes]::ReadOnly) {
   $msg = "  ⚠️ File is read-only."
   Write-Warning $msg
   $fileReport += $msg
   $issuesInFile++
  }
  if ($file.Extension -in ('.sh', '.py', '.ps1')) {
   try {
    $acl = Get-Acl $file.FullName
    $isExecutable = $false
    foreach ($ace in $acl.Access) {
     if ($ace.FileSystemRights -match 'ExecuteFile') {
      $isExecutable = $true
      break
     }
    }
    if (-not $isExecutable) {
     $msg = "  ⚠️ Script file may not be executable (no ExecuteFile permission)."
     Write-Warning $msg
     $fileReport += $msg
     $issuesInFile++
    }
   }
   catch {}
  }
 }

 if ($issuesInFile -gt 0) {
  $overallIssues += $issuesInFile
  $filesWithIssues++
  Write-Host "  Found $issuesInFile issue(s) in this file." -ForegroundColor Yellow
  $fileReport = @("$($file.FullName): $issuesInFile issue(s)") + $fileReport
  $reportLines += $fileReport
 }
 else {
  Write-Host "  ✅ No finalizer issues found." -ForegroundColor Green
  $reportLines += "$($file.FullName): OK"
 }
}

Write-Host "-------------------------------------"
Write-Host "📊 Finalizer Check Summary:" -ForegroundColor Cyan
Write-Host "Total Files Checked: $($filesToFinalize.Count)"
Write-Host "Files with Issues: $filesWithIssues"
Write-Host "Total Issues Found: $overallIssues"
Write-Host "Completed at: $(Get-Date)" -ForegroundColor Cyan

if ($ReportFile -and $ReportFile.Trim() -ne "") {
 $reportLines | Out-File -FilePath $ReportFile -Encoding UTF8
 Write-Host "Report written to $ReportFile" -ForegroundColor Cyan
}

if ($overallIssues -gt 0) {
 Write-Warning "Finalizer checks found issues. Please review the output above."
 exit 1
}
else {
 Write-Host "🎉 All files passed finalizer checks! Ready for distribution/commit." -ForegroundColor Green
 exit 0
}
