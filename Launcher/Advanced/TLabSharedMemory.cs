using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace TLabShareScreenWireless
{
    class TLabSharedMemory
    {
        void TermProcess()
        {
            MemoryMappedFile memoryMappedFile = null;
            MemoryMappedViewAccessor memoryMappedViewAccessor = null;
            string sharedMemoryName = "TLabShareScreenServer";

            try
            {
                memoryMappedFile = MemoryMappedFile.OpenExisting(sharedMemoryName);
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine(ex.Message);

                memoryMappedFile = MemoryMappedFile.CreateOrOpen(
                    sharedMemoryName,
                    Marshal.SizeOf<byte>()
                );

                Console.WriteLine("shared memory created");
            }

            if (memoryMappedFile == null)
            {
                Console.WriteLine("memoryMappedFile is null");
                return;
            }

            memoryMappedViewAccessor = memoryMappedFile.CreateViewAccessor();

            if (memoryMappedViewAccessor == null)
            {
                Console.WriteLine("memoryMappedViewAccessor is null");
                return;
            }

            while (true)
            {
                if (memoryMappedViewAccessor.CanWrite)
                {
                    memoryMappedViewAccessor.Write(0, (byte)0);
                    memoryMappedViewAccessor.Flush();
                    break;
                }
            }

            memoryMappedViewAccessor.Dispose();
            memoryMappedFile.Dispose();
        }
    }
}
