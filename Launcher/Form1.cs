using System;
using System.Windows.Forms;

namespace TLab.MTPEG
{
    public partial class Form1 : Form
    {
        private bool m_shareing = false;

        private const string MTPEG_SERVER = "MTPEGServer";

        private const string MTPEG_SERVER_KILL = "MTPEGServerKill";

        public Form1()
        {
            InitializeComponent();
        }

        private void ShareButton_Click(object sender, EventArgs e)
        {
            if(m_shareing)
            {
                Share_Stop();
                return;
            }

            ushort client_port = ushort.Parse(ClientPortText.Text);
            ushort server_port = ushort.Parse(ServerPortText.Text);

            if (!Util.CheckPortRange(client_port))
            {
                Console.WriteLine(string.Format("client port out of range: {0:d}", client_port));
                return;
            }

            if (!Util.CheckPortRange(server_port))
            {
                Console.WriteLine(string.Format("server port out of range: {0:d}", server_port));
                return;
            }

            string client_addr = ClientAddr.Text;

            if (!Util.CheckAddressPattern(client_addr))
            {
                Console.WriteLine("client addr: regex not matching");
                return;
            }

            Console.WriteLine("start set up client: " + client_addr.ToString());

            string args = Util.CombineArgs(new string[] {
                server_port.ToString(),
                client_port.ToString(),
                client_addr.ToString()
            });
            Util.StartProcess(MTPEG_SERVER, args);

            ShareButton.Text = "Stop";
            m_shareing = true;
        }

        private void Share_Stop()
        {
            Util.StartProcess(MTPEG_SERVER_KILL);

            m_shareing = false;
            ShareButton.Text = "Run";
        }

        private void Form1_Load(object sender, EventArgs e) { }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if(m_shareing)
            {
                Share_Stop();
            }
        }

        private void PortText_KeyPress(object sender, KeyPressEventArgs e)
        {
            // 49152 ~ 65535

            char[] num_array = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

            if (char.IsControl(e.KeyChar))  // 制御文字は入力可
            {
                e.Handled = false;
                return;
            }

            for (int i = 0; i < num_array.Length; i++)  // 数字(0-9)は入力可
            {
                if (e.KeyChar == num_array[i])
                {
                    e.Handled = false;
                    return;
                }
            }

            e.Handled = true;   // 上記以外は入力不可
        }
    }
}
