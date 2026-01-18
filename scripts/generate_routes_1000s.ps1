# Generate 1000s route files for fast testing
# Usage: .\scripts\generate_routes_1000s.ps1

$demands = @(600, 800, 900, 1000)
$seeds = @(42, 43)
$duration = 1000

# Create output directory
$outputDir = "networks\variants\train_1000s"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "Generating 1000s route files..." -ForegroundColor Cyan
Write-Host "Duration: ${duration}s, Seeds: $($seeds -join ', ')" -ForegroundColor Cyan
Write-Host ""

$total = $demands.Count * $seeds.Count
$current = 0

foreach ($demand in $demands) {
    foreach ($seed in $seeds) {
        $current++
        $outputFile = "$outputDir\bignet_train_seed$($seed.ToString('00000'))_d${demand}.rou.xml"
        
        Write-Host "[$current/$total] Generating d=$demand, seed=$seed..." -ForegroundColor Yellow
        
        python scripts/generate_jtr_data.py `
            --net-file networks/BIGNET.net.xml `
            --output-route $outputFile `
            --seed $seed `
            --base-flow $demand `
            --duration $duration
        
        if ($LASTEXITCODE -eq 0 -and (Test-Path $outputFile)) {
            $sizeKB = [math]::Round((Get-Item $outputFile).Length / 1KB, 1)
            Write-Host "  ✓ Created: $((Get-Item $outputFile).Name) (${sizeKB} KB)" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Failed to create route file" -ForegroundColor Red
        }
        Write-Host ""
    }
}

# Create manifests
Write-Host "Creating manifest files..." -ForegroundColor Cyan
foreach ($demand in $demands) {
    $manifestPath = "$outputDir\manifest_d${demand}.txt"
    $routes = @()
    foreach ($seed in $seeds) {
        $filename = "bignet_train_seed$($seed.ToString('00000'))_d${demand}.rou.xml"
        $routes += $filename
    }
    $routes | Out-File -FilePath $manifestPath -Encoding utf8
    Write-Host "  ✓ Created: manifest_d${demand}.txt" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done! Generated $total route files in $outputDir" -ForegroundColor Green
