@echo off
cd /d D:\fm_paper_works\hcc-vs-hemangioma
echo START %DATE% %TIME% > scratch\_vis_out.log
C:\Python314\python.exe scratch\vis_mimic_cam_attn.py >> scratch\_vis_out.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> scratch\_vis_out.log
echo DONE %DATE% %TIME% >> scratch\_vis_out.log
