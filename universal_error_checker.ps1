# Universal Error Checker
# Scans files for common error patterns and keywords.

param(
    [Parameter(Mandatory = $false, Position = 0)]
    [string]$Path = ".",

    [Parameter(Mandatory = $false)]
    [string[]]$FileTypes = @("log", "txt", "ps1", "py", "md"), # Common file types to check

    [Parameter(Mandatory = $false)]
    [switch]$Recurse,

    [Parameter(Mandatory = $false)]
    [string[]]$ErrorKeywords = @(
        "error", "exception", "fail", "failed", "fatal", "critical", "traceback", 
        "unhandled", "denied", "timeout", "refused", "not found", "cannot open", "unable to",
        "syntax error", "type error", "name error", "attribute error", "import error",
        "indentation error", "unicode error", "value error", "key error", "index error",
        "runtime error", "memory error", "overflow error", "zero division", "assertion error",
        "file not found", "permission denied", "access denied", "connection refused",
        "broken pipe", "operation timed out", "host unreachable", "network unreachable"
    ),

    [Parameter(Mandatory = $false)]
    [int]$ContextLines = 2, # Lines before and after the error keyword to display

    [Parameter(Mandatory = $false)]
    [switch]$CaseSensitive = $false,

    [Parameter(Mandatory = $false)]
    [switch]$ShowSummaryOnly = $false,

    [Parameter(Mandatory = $false)]
    [string]$OutputFile = "",

    [Parameter(Mandatory = $false)]
    [int]$MaxFileSize = 100MB # Skip files larger than this
)

Write-Host "🕵️ Starting Universal Error Check..." -ForegroundColor Cyan
Write-Host "Path: $Path" -ForegroundColor Cyan
Write-Host "File Types: $($FileTypes -join ', ')" -ForegroundColor Cyan
Write-Host "Recurse: $Recurse" -ForegroundColor Cyan
Write-Host "Error Keywords: $($ErrorKeywords.Count) keywords configured" -ForegroundColor Cyan
Write-Host "Context Lines: $ContextLines" -ForegroundColor Cyan
Write-Host "Case Sensitive: $CaseSensitive" -ForegroundColor Cyan
Write-Host "-------------------------------------"

$outputLines = @()
if ($OutputFile) {
    $outputLines += "Universal Error Check Report - $(Get-Date)"
    $outputLines += "=" * 50
}

$filesToScan = @()
if (Test-Path $Path -PathType Container) {
    $FileTypes | ForEach-Object {
        $ext = $_.TrimStart(".")
        $pattern = "*.$ext"
        try {
            $foundFiles = Get-ChildItem -Path $Path -Filter $pattern -Recurse:$Recurse -File -ErrorAction SilentlyContinue
            if ($foundFiles) {
                $filesToScan += $foundFiles
            }
        }
        catch {
            Write-Warning "Error searching for files with pattern $pattern`: $($_.Exception.Message)"
        }
    }
}
elseif (Test-Path $Path -PathType Leaf) {
    $filesToScan += Get-Item $Path
}
else {
    Write-Error "Path '$Path' not found."
    exit 1
}

if ($filesToScan.Count -eq 0) {
    Write-Host "No files found to scan with the specified criteria." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($filesToScan.Count) files to scan." -ForegroundColor Green

$totalErrorsFoundCount = 0
$filesWithErrorsCount = 0
$skippedFilesCount = 0
$fileResults = @()

foreach ($fileItem in $filesToScan | Sort-Object FullName -Unique) {
    Write-Host "Scanning: $($fileItem.FullName)" -ForegroundColor White
    
    # Check file size
    if ($fileItem.Length -gt $MaxFileSize) {
        Write-Warning "  Skipping large file ($([math]::Round($fileItem.Length / 1MB, 2)) MB)"
        $skippedFilesCount++
        continue
    }
    
    try {
        $fileContent = Get-Content $fileItem.FullName -ErrorAction Stop
    }
    catch {
        Write-Warning "  Could not read file: $($fileItem.FullName) - $($_.Exception.Message)"
        continue
    }

    if (-not $fileContent) {
        Write-Warning "  File is empty: $($fileItem.FullName)"
        continue
    }

    $errorsInFile = 0
    $lineNumber = 0
    $fileErrorDetails = @()
 
    foreach ($line in $fileContent) {
        $lineNumber++
        foreach ($keyword in $ErrorKeywords) {
            $matchFound = $false
            
            if ($CaseSensitive) {
                $matchFound = $line -cmatch [regex]::Escape($keyword)
            }
            else {
                $matchFound = $line -imatch [regex]::Escape($keyword)
            }
            
            if ($matchFound) {
                $errorsInFile++
                $totalErrorsFoundCount++
                
                # Create error detail object
                $errorDetail = @{
                    Keyword     = $keyword
                    LineNumber  = $lineNumber
                    LineContent = $line.Trim()
                    Context     = @()
                }
                
                # Collect context lines
                $startContext = [Math]::Max(0, $lineNumber - $ContextLines - 1)
                $endContext = [Math]::Min($fileContent.Count - 1, $lineNumber + $ContextLines - 1)
                
                for ($i = $startContext; $i -le $endContext; $i++) {
                    $displayLineNumber = $i + 1
                    $displayLine = $fileContent[$i]
                    $contextLine = @{
                        LineNumber  = $displayLineNumber
                        Content     = $displayLine
                        IsErrorLine = ($displayLineNumber -eq $lineNumber)
                    }
                    $errorDetail.Context += $contextLine
                }
                
                $fileErrorDetails += $errorDetail
                
                if (-not $ShowSummaryOnly) {
                    Write-Host "  ❌ Found keyword '$keyword' on line $lineNumber:" -ForegroundColor Red
                    
                    foreach ($contextLine in $errorDetail.Context) {
                        $lineFormat = "    {0:D4}: {1}" -f $contextLine.LineNumber, $contextLine.Content
                        if ($contextLine.IsErrorLine) {
                            Write-Host $lineFormat -ForegroundColor Yellow
                        }
                        else {
                            Write-Host $lineFormat -ForegroundColor Gray
                        }
                    }
                    Write-Host ""
                }
                
                # Add to output file if specified
                if ($OutputFile) {
                    $outputLines += ""
                    $outputLines += "File: $($fileItem.FullName)"
                    $outputLines += "Keyword: $keyword (Line $lineNumber)"
                    $outputLines += "Context:"
                    foreach ($contextLine in $errorDetail.Context) {
                        $prefix = if ($contextLine.IsErrorLine) { ">>> " } else { "    " }
                        $outputLines += "$prefix$($contextLine.LineNumber): $($contextLine.Content)"
                    }
                }
                
                # Move to next line once a keyword is found in the current line
                break
            }
        }
    }

    if ($errorsInFile -gt 0) {
        $filesWithErrorsCount++
        $fileResult = @{
            FilePath     = $fileItem.FullName
            ErrorCount   = $errorsInFile
            ErrorDetails = $fileErrorDetails
        }
        $fileResults += $fileResult
        
        if (-not $ShowSummaryOnly) {
            Write-Host "  Found $errorsInFile error keyword(s) in this file." -ForegroundColor Yellow
        }
    }
    else {
        if (-not $ShowSummaryOnly) {
            Write-Host "  ✅ No error keywords found." -ForegroundColor Green
        }
    }
}

Write-Host "-------------------------------------"
Write-Host "📊 Error Check Summary:" -ForegroundColor Cyan
Write-Host "Total Files Scanned: $($filesToScan.Count)"
Write-Host "Files with Error Keywords: $filesWithErrorsCount"
Write-Host "Total Error Keywords Found: $totalErrorsFoundCount"
if ($skippedFilesCount -gt 0) {
    Write-Host "Files Skipped (too large): $skippedFilesCount" -ForegroundColor Yellow
}

# Detailed summary by file
if ($ShowSummaryOnly -and $fileResults.Count -gt 0) {
    Write-Host ""
    Write-Host "Files with Errors:" -ForegroundColor Yellow
    foreach ($result in $fileResults | Sort-Object ErrorCount -Descending) {
        Write-Host "  $($result.FilePath): $($result.ErrorCount) error(s)" -ForegroundColor Red
    }
    
    # Top keywords
    $keywordCounts = @{}
    foreach ($result in $fileResults) {
        foreach ($error in $result.ErrorDetails) {
            if ($keywordCounts.ContainsKey($error.Keyword)) {
                $keywordCounts[$error.Keyword]++
            }
            else {
                $keywordCounts[$error.Keyword] = 1
            }
        }
    }
    
    if ($keywordCounts.Count -gt 0) {
        Write-Host ""
        Write-Host "Most Common Error Keywords:" -ForegroundColor Yellow
        $topKeywords = $keywordCounts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 10
        foreach ($kw in $topKeywords) {
            Write-Host "  $($kw.Name): $($kw.Value) occurrence(s)" -ForegroundColor Red
        }
    }
}

# Save output file if specified
if ($OutputFile -and $outputLines.Count -gt 0) {
    try {
        $outputLines | Out-File -FilePath $OutputFile -Encoding UTF8
        Write-Host "Report saved to: $OutputFile" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to save report to $OutputFile`: $($_.Exception.Message)"
    }
}

if ($totalErrorsFoundCount -gt 0) {
    Write-Warning "Potential errors found. Please review the output above."
    exit 1 # Indicate errors were found
}
else {
    Write-Host "🎉 No potential errors found in the scanned files!" -ForegroundColor Green
    exit 0
}
