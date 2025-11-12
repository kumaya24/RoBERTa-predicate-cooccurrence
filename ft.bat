@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM Automated Script for Fine-Tuning Experiments (Windows Batch)
REM This script iterates through lists of parameters and executes the
REM 'python run_ft.py' command for every combination.
REM NOTE: This script mirrors the configuration from the original .sh file.
REM ====================================================================

REM --- CONFIGURATION: Set the lists of values you want to test ---

REM Define the different task types
set TYPE_OPTIONS=agent_vintran agent_vtran evt_vintran evt_vtran participleAdj_vintran participleAdj_vtran

REM Define the number of epochs to run
set EPOCHS=6

REM Define the batch sizes
set BATCH_SIZES=8

echo Starting Automated Experiment Runner...
echo.

REM Loop 1: Iterate over TYPE_OPTIONS
for %%t in (%TYPE_OPTIONS%) do (
    set current_type=%%t
    echo ====================================================
    echo --- Starting Type: !current_type! ---
    echo ====================================================

    REM Loop 2: Iterate over EPOCHS
    for %%e in (%EPOCHS%) do (
        set current_epochs=%%e

        REM Loop 3: Iterate over BATCH_SIZES
        for %%b in (%BATCH_SIZES%) do (
            set current_batch_size=%%b

            REM --- CONSTRUCT THE COMMAND ---
            set "COMMAND=python run_ft.py !current_type! --epochs !current_epochs! --batch_size !current_batch_size! --model t5"

            echo.
            echo [NEW RUN]
            echo Configuration: Type=!current_type!, Epochs=!current_epochs!, Batch=!current_batch_size!
            echo Running: !COMMAND!
            echo.

            REM --- EXECUTE THE COMMAND ---
            REM CALL is used to execute the command and ensures the batch script waits for the python process
            CALL !COMMAND!

            REM Optional: Check for execution errors (Errorlevel 1 typically indicates a crash or error)
            if errorlevel 1 (
                echo !!! WARNING: Script failed for this run. Continuing to next combination. !!!
            )
        )
    )
)

echo.
echo ====================================================
echo All experiment combinations have been completed!
echo ====================================================

endlocal
pause
