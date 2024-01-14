# MTPEGServerKill

# output destination folder
- $(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
- $(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\

## debug
```
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe*"
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb*"
```

## release
```
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.exe*"
xcopy /y "$(SolutionDir)MTPEGServerKill\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\MTPEGServerKill.pdb*"
```