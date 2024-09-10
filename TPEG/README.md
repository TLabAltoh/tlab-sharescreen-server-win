# TPEG
This is a library for encoding and decoding bitmap frames with CUDA, compression algorithms based on JPEG without Huffman encoding.

## Output Destination Folder
```
$(SolutionDir)Launcher\bin\$(Platform)\$(Configuration)\
$(SolutionDir)TPEGDLLTest\bin\$(Platform)\$(Configuration)\
$(SolutionDir)TPEG\bin\$(Platform)\$(Configuration)\
```

## Build Event

### Release
```bat
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
