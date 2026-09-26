@echo off
rem Salted-figure regeneration chain (overnight dry brine defaults).
rem Sequential: fast demos first, then the planner-heavy trio; robust
rem warm-starts from the fresh fullplan result.
set PYTHONUTF8=1
cd /d "%~dp0..\app"
set LOG=..\data\regen_logs
echo [%date% %time%] start salted forward_demo > %LOG%\salted_driver.log
call uv run python -u ..\scripts\forward_demo.py > %LOG%\salted_forward.log 2>&1
echo [%date% %time%] start salted bark >> %LOG%\salted_driver.log
call uv run python -u ..\scripts\run_bark.py > %LOG%\salted_bark.log 2>&1
echo [%date% %time%] start salted control >> %LOG%\salted_driver.log
call uv run python -u ..\scripts\run_control.py > %LOG%\salted_control.log 2>&1
echo [%date% %time%] start salted fullplan >> %LOG%\salted_driver.log
call uv run python -u ..\scripts\run_fullplan.py > %LOG%\salted_fullplan.log 2>&1
echo [%date% %time%] start salted mpc >> %LOG%\salted_driver.log
call uv run python -u ..\scripts\run_mpc.py > %LOG%\salted_mpc.log 2>&1
echo [%date% %time%] start salted robust >> %LOG%\salted_driver.log
call uv run python -u ..\scripts\run_robust_control.py > %LOG%\salted_robust.log 2>&1
echo [%date% %time%] salted regeneration done >> %LOG%\salted_driver.log
