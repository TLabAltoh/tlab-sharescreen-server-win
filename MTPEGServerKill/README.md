# MTPEGServerKill

## Output Destination Folder
```
$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\
```

## Build Event
### Debug
```bat
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe*"
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb*"
```

### Release
```bat
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe*"
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb*"
```
