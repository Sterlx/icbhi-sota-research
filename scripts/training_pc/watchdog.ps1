# ============================================================
# Training PC Watchdog — Windows PowerShell Version
# Polls GitHub for new experiments, runs training, pushes results
# ============================================================
# Usage: .\scripts\training_pc\watchdog.ps1
# ============================================================

param(
    [int]$PollIntervalSeconds = 60,
    [string]$PythonExe = "python"  # or path to venv python
)

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ICBHI Training Watchdog (Windows)" -ForegroundColor Cyan
Write-Host "  Project: $ProjectDir" -ForegroundColor Cyan
Write-Host "  Time: $(Get-Date)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Checking for new experiments..."

    # Pull latest from GitHub
    git pull origin main 2>$null

    # Check for trigger file
    if (Test-Path "experiments\TRIGGER.yaml") {
        try {
            $triggerContent = Get-Content "experiments\TRIGGER.yaml" -Raw
            # Simple check for "status: queued"
            if ($triggerContent -match "status:\s*queued") {
                # Extract experiment ID
                $expIdMatch = [regex]::Match($triggerContent, "experiment_id:\s*(\S+)")
                $ExpId = if ($expIdMatch.Success) { $expIdMatch.Groups[1].Value } else { "unknown" }

                Write-Host ""
                Write-Host ("=" * 60)
                Write-Host "  STARTING experiment: $ExpId"
                Write-Host ("=" * 60)

                # Update status to running
                $triggerContent = $triggerContent -replace "status:\s*queued", "status: running"
                $triggerContent += "`nstarted_at: $(Get-Date -Format 'o')"
                Set-Content "experiments\TRIGGER.yaml" $triggerContent

                git add experiments\TRIGGER.yaml
                git commit -m "[STATUS] Running $ExpId" 2>$null
                git push origin main 2>$null

                # Find config file
                $ConfigFile = "experiments\$ExpId\config.yaml"
                if (-not (Test-Path $ConfigFile)) {
                    $ConfigFile = Get-ChildItem "configs\models\*.yaml" | Select-Object -First 1
                }

                if ($ConfigFile) {
                    Write-Host "   Config: $ConfigFile"
                    Write-Host "   Training..."
                    
                    & $PythonExe src\training\train.py --config $ConfigFile --exp-id $ExpId
                    
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "   Evaluating..."
                        & $PythonExe src\evaluation\evaluate.py --exp-id $ExpId
                        
                        # Mark as completed
                        $triggerContent = Get-Content "experiments\TRIGGER.yaml" -Raw
                        $triggerContent = $triggerContent -replace "status:\s*running", "status: completed"
                        $triggerContent += "`ncompleted_at: $(Get-Date -Format 'o')"
                        Set-Content "experiments\TRIGGER.yaml" $triggerContent
                        
                        Write-Host "   Pushing results..."
                        git add "experiments\$ExpId\" experiments\TRIGGER.yaml results\
                        git commit -m "[RESULTS] $ExpId completed" 2>$null
                        git push origin main 2>$null
                        
                        Write-Host "   DONE: Experiment $ExpId complete!" -ForegroundColor Green
                    } else {
                        Write-Host "   TRAINING FAILED!" -ForegroundColor Red
                        $triggerContent = Get-Content "experiments\TRIGGER.yaml" -Raw
                        $triggerContent = $triggerContent -replace "status:\s*running", "status: failed"
                        Set-Content "experiments\TRIGGER.yaml" $triggerContent
                        git add experiments\TRIGGER.yaml
                        git commit -m "[FAILED] $ExpId" 2>$null
                        git push origin main 2>$null
                    }
                } else {
                    Write-Host "   ERROR: No config file found!" -ForegroundColor Red
                }
                Write-Host ""
            }
        } catch {
            Write-Host "   Error reading trigger: $_" -ForegroundColor Red
        }
    }

    # Wait before next poll
    Start-Sleep -Seconds $PollIntervalSeconds
}
