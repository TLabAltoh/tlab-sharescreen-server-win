using System;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace TLabShareScreenWireless
{
    public partial class Form1 : Form
    {
        private bool shareing = false;

        public Form1()
        {
            InitializeComponent();
        }

        private void ShareButton_Click(object sender, EventArgs e)
        {
            if(shareing == true)
            {
                Share_Stop();
                return;
            }

            ushort cport = ushort.Parse(ClientPortText.Text);
            ushort sport = ushort.Parse(ServerPortText.Text);

            if (cport < (ushort)49152 || cport > (ushort)65535)
            {
                Console.WriteLine(string.Format("client port out of range: {0:d}", cport));
                return;
            }

            if (sport < (ushort)49152 || sport > (ushort)65535)
            {
                Console.WriteLine(string.Format("server port out of range: {0:d}", sport));
                return;
            }

            string caddr = ClientAddr.Text;

            // \d{1, 3}: 1~3桁の数である
            // (\.\d{1,3}){3}: ピリオドと 1~3桁の数の組み合わせが 3回続く
            if (Regex.IsMatch(caddr, @"^\d{1,3}(\.\d{1,3}){3}$") == false)
            {
                Console.WriteLine("client addr: regex not matching");
                return;
            }

            Console.WriteLine("start set up client: " + caddr.ToString());

            ProcessStartInfo server = new ProcessStartInfo();
            server.FileName = "TLabShareScreenServer";
            server.Arguments = sport.ToString() + " " + cport.ToString() + " " + caddr.ToString();
            Process.Start(server);

            ShareButton.Text = "Stop";
            shareing = true;
        }

        private void Share_Stop()
        {
            ProcessStartInfo server = new ProcessStartInfo();
            server.FileName = "TLabShareScreenServerKill";
            Process.Start(server);

            shareing = false;
            ShareButton.Text = "Run";
        }

        private void Form1_Load(object sender, EventArgs e) { }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if(shareing == true) Share_Stop();
        }

        private void PortText_KeyPress(object sender, KeyPressEventArgs e)
        {
            // 49152 ~ 65535

            char[] numArray = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

            // 制御文字は入力可
            if (char.IsControl(e.KeyChar))
            {
                e.Handled = false;
                return;
            }

            // 数字(0-9)は入力可
            for (int i = 0; i < numArray.Length; i++)
            {
                if (e.KeyChar == numArray[i])
                {
                    e.Handled = false;
                    return;
                }
            }

            // 上記以外は入力不可
            e.Handled = true;
        }
    }
}
