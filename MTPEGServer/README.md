# MTPEGServer

# output destination folder
- $(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
- $(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\

## debug
```
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.exe*"
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb*"
```

## release
```
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.exe*"
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb*"
```