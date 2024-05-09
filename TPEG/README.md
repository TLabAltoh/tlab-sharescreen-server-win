# TPEG
A library for frame encoding and decoding using CUDA, implemented based on JPEG or without Huffman encoding.

- use prject/property/detail/character_set mbcs

# output destination folder
- $(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
- $(SolutionDir)TPEGDLLTest\bin\$(Platform)\$(Configuration)\
- $(SolutionDir)TPEG\bin\$(Platform)\$(Configuration)\

# after build event
- [xcopy if directory no exist](https://superuser.com/questions/119263/how-to-copy-a-file-to-a-directory-in-dos-and-create-directories-if-necessary)
- [xcopy refarence](https://learn.microsoft.com/ja-jp/windows-server/administration/windows-commands/xcopy)

## debug
```
```

## release
```
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)Launcher\bin\$(Platform)\Debug\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)Launcher\bin\$(Platform)\Debug\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)Launcher\bin\$(Platform)\Debug\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\Debug\TPEG.pdb*"

xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)Launcher\bin\$(Platform)\Release\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)Launcher\bin\$(Platform)\Release\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)Launcher\bin\$(Platform)\Release\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)Launcher\bin\$(Platform)\Release\TPEG.pdb*"

xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Debug\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Debug\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Debug\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Debug\TPEG.pdb*"

xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Release\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Release\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Release\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)MTPEGServer\bin\$(Platform)\Release\TPEG.pdb*"

xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Debug\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Debug\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Debug\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Debug\TPEG.pdb*"

xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.dll" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Release\TPEG.dll*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.exp" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Release\TPEG.exp*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.lib" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Release\TPEG.lib*"
xcopy /y "$(SolutionDir)TPEG\bin\$(Platform)\Release\TPEG.pdb" "$(SolutionDir)TPEGDLLTest\bin\$(Platform)\Release\TPEG.pdb*"
```