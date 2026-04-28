# DAQ 장비 운용 중 Windows 절전/화면보호기로 인한 PXI 연결 끊김 방지
# 관리자 권한으로 실행 필요

Write-Host "=== DAQ 전원 관리 설정 ===`n" -ForegroundColor Cyan

# 고성능 전원 계획 활성화
powercfg /setactive SCHEME_MIN
Write-Host "[OK] 전원 계획: 고성능" -ForegroundColor Green

# 모니터 끄기 비활성화 (AC)
powercfg /change monitor-timeout-ac 0
Write-Host "[OK] 모니터 끄기: 사용 안 함" -ForegroundColor Green

# 시스템 절전 비활성화 (AC)
powercfg /change standby-timeout-ac 0
Write-Host "[OK] 시스템 절전: 사용 안 함" -ForegroundColor Green

# 화면보호기 비활성화 (레지스트리)
Set-ItemProperty -Path "HKCU:\Control Panel\Desktop" -Name "ScreenSaveActive" -Value "0"
Write-Host "[OK] 화면보호기: 사용 안 함" -ForegroundColor Green

# USB 선택적 일시 중단 비활성화
powercfg /setacvalueindex SCHEME_MIN SUB_USB USBSELECTIVE 0
Write-Host "[OK] USB 선택적 일시 중단: 사용 안 함" -ForegroundColor Green

# PCI Express 링크 상태 전원 관리 비활성화 (PXI 버스 안정성)
powercfg /setacvalueindex SCHEME_MIN SUB_PCIEXPRESS ASPM 0
Write-Host "[OK] PCI Express 링크 전원 관리: 사용 안 함" -ForegroundColor Green

powercfg /setactive SCHEME_MIN

Write-Host "`n설정 완료. 재시작 없이 즉시 적용됩니다." -ForegroundColor Cyan
