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
 [switch]$EnsureFinalNewline
)

Write-Host "🧐 Starting File Finalizer Check..." -ForegroundColor Cyan
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

$overallIssues = 0
$filesWithIssues = 0

foreach ($file in $filesToFinalize | Sort-Object -Unique) {
 Write-Host "Finalizing: $($file.FullName)" -ForegroundColor White
 $issuesInFile = 0

 # 1. Check File Size
 if (($file.Length / 1KB) -gt $MaxSizeKB) {
  Write-Warning "  ⚠️ File size ($($file.Length / 1KB) KB) exceeds maximum ($MaxSizeKB KB)."
  $issuesInFile++
 }

 # Read content for other checks
 $fileContentBytes = Get-Content $file.FullName -AsByteStream -Raw -ErrorAction SilentlyContinue
 if (-not $fileContentBytes) {
  Write-Warning "  Could not read file or file is empty: $($file.FullName)"
  # Allow empty files, but skip content-based checks
 }
 else {
  $fileContentString = Get-Content $file.FullName -Raw -Encoding Default # Read as string for keyword search

  # 2. Check for Forbidden Keywords
  foreach ($keyword in $ForbiddenKeywords) {
   if ($fileContentString -match $keyword) {
    Write-Warning "  ⚠️ Found forbidden keyword '$keyword'."
    $issuesInFile++
    # Could add line number reporting here if needed
   }
  }

  # 3. Check for Trailing Whitespace
  if ($CheckTrailingWhitespace) {
   $lines = Get-Content $file.FullName -Encoding Default
   for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "\s+$") {
     Write-Warning "  ⚠️ Trailing whitespace found on line $($i+1)."
     $issuesInFile++
     break # Report once per file for this check
    }
   }
  }

  # 4. Check for UTF-8 BOM
  if ($CheckUtf8Bom) {
   if ($fileContentBytes.Count -ge 3 -and $fileContentBytes[0] -eq 0xEF -and $fileContentBytes[1] -eq 0xBB -and $fileContentBytes[2] -eq 0xBF) {
    Write-Warning "  ⚠️ File contains UTF-8 BOM."
    $issuesInFile++
   }
  }
        
  # 5. Ensure Final Newline
  if ($EnsureFinalNewline) {
   if ($fileContentBytes.Count -gt 0 -and $fileContentBytes[-1] -ne 0x0A) {
    # LF
    # Also check for CRLF if needed, but LF is common for final newline
    if ($fileContentBytes.Count -lt 2 -or ($fileContentBytes[-2] -ne 0x0D -or $fileContentBytes[-1] -ne 0x0A)) {
     # Not CRLF
     Write-Warning "  ⚠️ File does not end with a newline character."
     $issuesInFile++
    }
   }
  }
 }

 if ($issuesInFile -gt 0) {
  $overallIssues += $issuesInFile
  $filesWithIssues++
  Write-Host "  Found $issuesInFile issue(s) in this file." -ForegroundColor Yellow
 }
 else {
  Write-Host "  ✅ No finalizer issues found." -ForegroundColor Green
 }
}

Write-Host "-------------------------------------"
Write-Host "📊 Finalizer Check Summary:" -ForegroundColor Cyan
Write-Host "Total Files Checked: $($filesToFinalize.Count)"
Write-Host "Files with Issues: $filesWithIssues"
Write-Host "Total Issues Found: $overallIssues"

if ($overallIssues -gt 0) {
 Write-Warning "Finalizer checks found issues. Please review the output above."
 exit 1 # Indicate issues were found
}
else {
 Write-Host "🎉 All files passed finalizer checks! Ready for distribution/commit." -ForegroundColor Green
 exit 0
}
