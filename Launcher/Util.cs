using System.Diagnostics;
using System.Text.RegularExpressions;

namespace TLab.MTPEG
{
    public static class Util
    {
        public static bool CheckPortRange(ushort port, ushort min = 49152, ushort max = 65535)
        {
            return !(port < min || port > max);
        }

        public static bool CheckAddressPattern(string addr)
        {
            // \d{1, 3}: 1~3桁の数である
            // (\.\d{1,3}){3}: ピリオドと 1~3桁の数の組み合わせが 3回続く
            return Regex.IsMatch(addr, @"^\d{1,3}(\.\d{1,3}){3}$");
        }

        public static Process StartProcess(string process_name, string args = null)
        {
            ProcessStartInfo process = new ProcessStartInfo();
            process.FileName = process_name;

            if (args != null)
            {
                process.Arguments = args;
            }

            return Process.Start(process);
        }

        public static string CombineArgs(string[] arg_array, string split = " ")
        {
            string args = "";

            for (int i = 0; i < arg_array.Length; i++)
            {
                args += arg_array[i];

                if (i < arg_array.Length - 1)
                {
                    args += split;
                }
            }

            return args;
        }
    }
}
