namespace TLabShareScreenWireless
{
    partial class Form1
    {
        /// <summary>
        /// 必要なデザイナー変数です。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        /// <param name="disposing">マネージド リソースを破棄する場合は true を指定し、その他の場合は false を指定します。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows フォーム デザイナーで生成されたコード

        /// <summary>
        /// デザイナー サポートに必要なメソッドです。このメソッドの内容を
        /// コード エディターで変更しないでください。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.ShareButton = new System.Windows.Forms.Button();
            this.ClientPortLabel = new System.Windows.Forms.TextBox();
            this.ClientPortText = new System.Windows.Forms.TextBox();
            this.ServerPortText = new System.Windows.Forms.TextBox();
            this.ServerPortLabel = new System.Windows.Forms.TextBox();
            this.ClientAddr = new System.Windows.Forms.TextBox();
            this.ClientAddrLabel = new System.Windows.Forms.TextBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // ShareButton
            // 
            this.ShareButton.Location = new System.Drawing.Point(9, 17);
            this.ShareButton.Margin = new System.Windows.Forms.Padding(2);
            this.ShareButton.Name = "ShareButton";
            this.ShareButton.Size = new System.Drawing.Size(94, 33);
            this.ShareButton.TabIndex = 0;
            this.ShareButton.Text = "Run";
            this.ShareButton.UseVisualStyleBackColor = true;
            this.ShareButton.Click += new System.EventHandler(this.ShareButton_Click);
            // 
            // ClientPortLabel
            // 
            this.ClientPortLabel.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.ClientPortLabel.Location = new System.Drawing.Point(471, 13);
            this.ClientPortLabel.Name = "ClientPortLabel";
            this.ClientPortLabel.ReadOnly = true;
            this.ClientPortLabel.Size = new System.Drawing.Size(120, 12);
            this.ClientPortLabel.TabIndex = 4;
            this.ClientPortLabel.Text = "CPort (49152~65535)";
            this.ClientPortLabel.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // ClientPortText
            // 
            this.ClientPortText.Location = new System.Drawing.Point(471, 31);
            this.ClientPortText.Name = "ClientPortText";
            this.ClientPortText.Size = new System.Drawing.Size(120, 19);
            this.ClientPortText.TabIndex = 5;
            this.ClientPortText.Text = "50000";
            this.ClientPortText.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.PortText_KeyPress);
            // 
            // ServerPortText
            // 
            this.ServerPortText.Location = new System.Drawing.Point(345, 31);
            this.ServerPortText.Name = "ServerPortText";
            this.ServerPortText.Size = new System.Drawing.Size(120, 19);
            this.ServerPortText.TabIndex = 7;
            this.ServerPortText.Text = "55555";
            this.ServerPortText.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.PortText_KeyPress);
            // 
            // ServerPortLabel
            // 
            this.ServerPortLabel.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.ServerPortLabel.Location = new System.Drawing.Point(345, 13);
            this.ServerPortLabel.Name = "ServerPortLabel";
            this.ServerPortLabel.ReadOnly = true;
            this.ServerPortLabel.Size = new System.Drawing.Size(120, 12);
            this.ServerPortLabel.TabIndex = 6;
            this.ServerPortLabel.Text = "SPort(49152~65535)";
            this.ServerPortLabel.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // ClientAddr
            // 
            this.ClientAddr.Location = new System.Drawing.Point(219, 31);
            this.ClientAddr.Name = "ClientAddr";
            this.ClientAddr.Size = new System.Drawing.Size(120, 19);
            this.ClientAddr.TabIndex = 9;
            this.ClientAddr.Text = "192.168.3.25";
            // 
            // ClientAddrLabel
            // 
            this.ClientAddrLabel.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.ClientAddrLabel.Location = new System.Drawing.Point(219, 13);
            this.ClientAddrLabel.Name = "ClientAddrLabel";
            this.ClientAddrLabel.ReadOnly = true;
            this.ClientAddrLabel.Size = new System.Drawing.Size(120, 12);
            this.ClientAddrLabel.TabIndex = 8;
            this.ClientAddrLabel.Text = "ClientAddr";
            this.ClientAddrLabel.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // pictureBox1
            // 
            this.pictureBox1.ImageLocation = "pixtureBox.png";
            this.pictureBox1.Location = new System.Drawing.Point(9, 60);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(582, 290);
            this.pictureBox1.TabIndex = 10;
            this.pictureBox1.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.AppWorkspace;
            this.ClientSize = new System.Drawing.Size(600, 360);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.ClientAddr);
            this.Controls.Add(this.ClientAddrLabel);
            this.Controls.Add(this.ServerPortText);
            this.Controls.Add(this.ServerPortLabel);
            this.Controls.Add(this.ClientPortText);
            this.Controls.Add(this.ClientPortLabel);
            this.Controls.Add(this.ShareButton);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "Form1";
            this.Text = "Form1";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button ShareButton;
        private System.Windows.Forms.TextBox ClientPortLabel;
        private System.Windows.Forms.TextBox ClientPortText;
        private System.Windows.Forms.TextBox ServerPortText;
        private System.Windows.Forms.TextBox ServerPortLabel;
        private System.Windows.Forms.TextBox ClientAddr;
        private System.Windows.Forms.TextBox ClientAddrLabel;
        private System.Windows.Forms.PictureBox pictureBox1;
    }
}

