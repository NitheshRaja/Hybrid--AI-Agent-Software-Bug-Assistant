# PostgreSQL Installation Verification Script

Write-Host "Checking PostgreSQL installation..." -ForegroundColor Cyan

# Check if psql is in PATH
$psqlPath = Get-Command psql -ErrorAction SilentlyContinue
if ($psqlPath) {
    Write-Host "✓ psql found at: $($psqlPath.Source)" -ForegroundColor Green
    $version = & psql --version
    Write-Host "  Version: $version" -ForegroundColor Green
} else {
    Write-Host "✗ psql not found in PATH" -ForegroundColor Red
    Write-Host "  You may need to add PostgreSQL bin directory to PATH:" -ForegroundColor Yellow
    Write-Host "  C:\Program Files\PostgreSQL\16\bin" -ForegroundColor Yellow
}

# Check if PostgreSQL service is running
$service = Get-Service -Name postgresql* -ErrorAction SilentlyContinue
if ($service) {
    Write-Host "`nPostgreSQL Services found:" -ForegroundColor Cyan
    foreach ($svc in $service) {
        $status = if ($svc.Status -eq 'Running') { "✓ Running" } else { "✗ $($svc.Status)" }
        $color = if ($svc.Status -eq 'Running') { "Green" } else { "Red" }
        Write-Host "  $($svc.Name): $status" -ForegroundColor $color
    }
} else {
    Write-Host "`n✗ No PostgreSQL service found" -ForegroundColor Red
}

Write-Host "`nTo test connection, run:" -ForegroundColor Cyan
Write-Host "  psql -U postgres" -ForegroundColor Yellow
Write-Host "`nThen enter the password you set during installation." -ForegroundColor Yellow











