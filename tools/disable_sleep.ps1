# DAQ power management configuration
# Prevents PXI bus disconnect caused by Windows sleep/screensaver
# Run as Administrator

Write-Host "=== DAQ Power Management Setup ===" -ForegroundColor Cyan

# High performance power plan
powercfg /setactive SCHEME_MIN
Write-Host "[OK] Power plan: High Performance" -ForegroundColor Green

# Disable monitor timeout (AC)
powercfg /change monitor-timeout-ac 0
Write-Host "[OK] Monitor timeout: Disabled" -ForegroundColor Green

# Disable system sleep (AC)
powercfg /change standby-timeout-ac 0
Write-Host "[OK] System sleep: Disabled" -ForegroundColor Green

# Disable screensaver (registry)
Set-ItemProperty -Path "HKCU:\Control Panel\Desktop" -Name "ScreenSaveActive" -Value "0"
Write-Host "[OK] Screensaver: Disabled" -ForegroundColor Green

# Disable USB selective suspend
powercfg /setacvalueindex SCHEME_MIN SUB_USB USBSELECTIVE 0
Write-Host "[OK] USB selective suspend: Disabled" -ForegroundColor Green

# Disable PCI Express link state power management (critical for PXI bus stability)
powercfg /setacvalueindex SCHEME_MIN SUB_PCIEXPRESS ASPM 0
Write-Host "[OK] PCI Express link power management: Disabled" -ForegroundColor Green

powercfg /setactive SCHEME_MIN

Write-Host "`nDone. Settings applied immediately." -ForegroundColor Cyan
