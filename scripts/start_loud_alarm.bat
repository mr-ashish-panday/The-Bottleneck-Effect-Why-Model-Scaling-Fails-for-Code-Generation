@echo off
title Loud Alarm
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0loud_alarm.ps1" %*
