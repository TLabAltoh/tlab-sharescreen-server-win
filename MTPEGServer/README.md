# MTPEGServer

## Output Destination Folder
```
$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\
```

## Build Event

### Debug
```bat
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.exe*"
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb*"
```

### Release
```bat
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.exe*"
xcopy /y "$(SolutionDir)MTPEGServer\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServer.pdb*"
```
