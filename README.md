# tlab-sharescreen-server-win
Software frame encoder using CUDA and cast encoded frames over UDP. Trying to implement a custom streaming protocol and shader based frame encoder/decoder for screencast.

> [!WARNING]  
> This is an experimental project and is only a degraded version of MPEG. Currently not intended for practical use. I hope you take it as an example of video streaming via custom protocol and software encoder/decoder.

## Screenshot (with use of [tlab-sharescreen-client-unity](https://github.com/TLabAltoh/tlab-sharescreen-client-unity))

### localhost (127.0.0.1)

<video src="https://user-images.githubusercontent.com/121733943/210447171-dd79dcfd-c64e-460e-81b2-7078929e0ea3.mp4"></video>

### Android Device

![DSC_0002](https://user-images.githubusercontent.com/121733943/211289979-46bfc2f3-c247-4015-b21d-ba5839f11a41.JPG)

## Operating Environment
| Property | Value                   |
| -------- | ----------------------- |
| OS       | Windows 10 Pro          |
| GPU      | NVIDIA GeForce RTX 3070 |
| CUDA     | 11.8                    |

## Client Software
- [unity](https://github.com/TLabAltoh/tlab-sharescreen-client-unity)