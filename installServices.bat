@echo off
setlocal

pushd expressapp
CALL npm install
popd

pushd reactapp
CALL npm install
popd

pushd Python
IF NOT EXIST venv (
    CALL python -m venv venv
)
pushd venv\Scripts
CALL activate.bat
popd
CALL pip install -r requirments.txt
popd

endlocal