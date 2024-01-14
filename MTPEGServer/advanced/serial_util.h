#pragma once

#define ACTIVE false

#if ACTIVE

#pragma comment(lib, "setupapi.lib")

#include <vector>
#include <string>
#include <iostream>
#include <Windows.h>
#include <SetupAPI.h>

using Tstring = std::wstring;
using Tchar = wchar_t;
using namespace std;

#define NO_DEVICE L"no device"
#define PATH L"\\\\.\\"
#define PORTS L"Ports"
#define PORTNAME L"PortName"

#pragma region CodeExample
// https://qiita.com/shunEnshuu/items/a9b0828aa33131b8e50f
#pragma endregion

class SerialInfo {

private:

	Tstring port_name;
	Tstring device;

public:

	const Tstring& port() const {
		// ポート名
		return port_name;
	}

	const Tstring& device_name() const {
		// フレンドリーネーム
		return device;
	}

	SerialInfo() {
		port_name = NO_DEVICE;
		device = NO_DEVICE;
	}

	#pragma region port_name()
	// コンストラクタを呼び出している
	// https://cpprefjp.github.io/lang/cpp11/uniform_initialization.html#:~:text=%E3%83%88%E3%83%A9%E3%82%AF%E3%82%BF%E5%91%BC%E3%81%B3%E5%87%BA%E3%81%97%0A%20%20X-,x1(0),-%3B%0A%20%20X%20x2
	#pragma endregion
	SerialInfo(const SerialInfo& info) :
		port_name(info.port_name),
		device(info.device) {
	}

	SerialInfo(const Tstring& _port) :
		port_name(_port) {
		auto list = getSerialList();
		for (auto dev : list) {
			if (dev.port() == port_name) {
				device = dev.device;
				return;
			}
		}
		port_name = NO_DEVICE;
		device = NO_DEVICE;
	}

	SerialInfo(
		const Tstring& _port,
		const Tstring& _device_name
	) :
		port_name(_port),
		device(_device_name) {
	}
};

std::vector<SerialInfo> getSerialList() {
	std::vector<SerialInfo> list;
	HDEVINFO hinfo = NULL;
	SP_DEVINFO_DATA info_data = { 0 };
	info_data.cbSize = sizeof(SP_DEVINFO_DATA);

	#pragma region GUID
	// https://ja.wikipedia.org/wiki/GUID
	#pragma endregion
	GUID guid;
	unsigned long guid_size = 0;

	#pragma region SetupDiClassGuidsFromName
	// https://learn.microsoft.com/ja-jp/windows/win32/api/setupapi/nf-setupapi-setupdiclassguidsfromnamea
	#pragma endregion
	if (
		SetupDiClassGuidsFromName(
			PORTS,
			&guid,
			1,
			&guid_size
		) == FALSE) {
		return list;
	}

	#pragma region SetupDiGetClassDevs
	// https://wiki.onakasuita.org/pukiwiki/?SetupDiGetClassDevs#:~:text=%E6%8C%87%E5%AE%9A%E3%81%95%E3%82%8C%E3%81%9F%E3%82%AF%E3%83%A9%E3%82%B9%E3%81%AB%E6%89%80%E5%B1%9E%E3%81%99%E3%82%8B%E3%81%99%E3%81%B9%E3%81%A6%E3%81%AE%E3%83%87%E3%83%90%E3%82%A4%E3%82%B9%E3%81%8C%E5%90%AB%E3%81%BE%E3%82%8C%E3%81%A6%E3%81%84%E3%82%8B1%E3%81%A4%E3%81%AE%E3%83%87%E3%83%90%E3%82%A4%E3%82%B9%E6%83%85%E5%A0%B1%E3%82%BB%E3%83%83%E3%83%88%E3%82%92%E8%BF%94%E3%81%97%E3%81%BE%E3%81%99%E3%80%82
	#pragma endregion
	hinfo = SetupDiGetClassDevs(
		&guid,
		0,
		0,
		DIGCF_PRESENT |
		DIGCF_PROFILE
	);

	if (hinfo == INVALID_HANDLE_VALUE) {
		return list;
	}

	Tchar buff[MAX_PATH];
	Tstring name;
	Tstring fullname;
	unsigned int index = 0;

	while (
		SetupDiEnumDeviceInfo(
			hinfo,
			index,
			&info_data
		)) {
		unsigned long type;
		unsigned long size;

		if (
			SetupDiGetDeviceRegistryProperty(
				hinfo,
				&info_data,
				SPDRP_DEVICEDESC,
				&type,
				(PBYTE)buff,
				MAX_PATH,
				&size
			) == TRUE) {
			// フレンドリーネーム(fullname)
			fullname = buff;
		}

		// とりあえず開く
		HKEY hkey = SetupDiOpenDevRegKey(
			hinfo,
			&info_data,
			DICS_FLAG_GLOBAL,
			0,
			DIREG_DEV, KEY_READ
		);

		if (hkey) {

			// ポート名取得
			RegQueryValueEx(
				hkey,
				PORTNAME,
				0,
				&type,
				(LPBYTE)buff,
				&size
			);

			// クローズ
			RegCloseKey(hkey);
			name = buff;
		}

		list.push_back(SerialInfo(name, fullname));
		index++;
	}

	SetupDiDestroyDeviceInfoList(hinfo);

	return list;
}

class Serial {

public:

	struct Config {
		unsigned int baudRate;
		unsigned int byteSize;

		enum class Parity {
			NO,
			EVEN,
			ODD
		} parity;

		enum class StopBits {
			ONE,
			ONE5,
			TWO
		} stopBits;
	};

private:

	SerialInfo info;
	Config conf;
	bool opened;
	void* handle;

	const Config defconf = {
		9600,
		8,
		Config::Parity::NO,
		Config::StopBits::ONE,
	};

	void setBuffSize(
		unsigned long read,
		unsigned long write
	) {
		SetupComm(handle, read, write);
	}

public:

	Serial() {
		conf = defconf;
		opened = false;
		handle = nullptr;
	}

	Serial(const Serial&) = delete;

	~Serial() {
		close();
	}

	bool open(
		const Tstring& port,
		unsigned int baudRate = 9600
	) {
		//デバイスをオープン

		if (opened) {
			return true;
		}

		info = SerialInfo(port);
		return open(info, baudRate);
	}

	bool open(
		const SerialInfo& serial_info,
		unsigned int baudRate = 9600
	) {
		info = serial_info;

		if (info.port() == NO_DEVICE) {
			return false;
		}

		// オープン
		Tstring path = PATH + info.port();
		handle = CreateFile(
			path.c_str(),
			GENERIC_READ | GENERIC_WRITE,
			0,
			NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			NULL
		);

		if (handle == INVALID_HANDLE_VALUE) {
			opened = false;
			return false;
		}

		conf.baudRate = baudRate;

		// 設定を反映
		setConfig(conf);

		// バッファサイズの決定
		setBuffSize(1024, 1024);

		opened = true;
		return true;
	}

	void close() {
		// デバイスをクローズ
		if (opened) {
			CloseHandle(handle);
		}
		opened = false;
	}

	const Config& getConfig() const {
		#pragma region const
		// constキーワードがつくとメンバ変数を変更できなくなる．
		#pragma endregion
		// ポート情報の取得
		return conf;
	}

	const SerialInfo& getInfo() const {
		// デバイス情報の取得
		return info;
	}

	void setConfig(const Config& cfg) {
		// DCB(ポート情報)を設定

		// DCBの設定を取得
		#pragma region DCB
		// https://learn.microsoft.com/ja-jp/windows/win32/devio/modification-of-communications-resource-settings#:~:text=DCB%20%E6%A7%8B%E9%80%A0%E4%BD%93%E3%81%AE%E3%83%A1%E3%83%B3%E3%83%90%E3%83%BC%E3%81%AF%E3%80%81%E3%83%9C%E3%83%BC%20%E3%83%AC%E3%83%BC%E3%83%88%E3%80%81%E3%83%90%E3%82%A4%E3%83%88%E3%81%82%E3%81%9F%E3%82%8A%E3%81%AE%E3%83%87%E3%83%BC%E3%82%BF%20%E3%83%93%E3%83%83%E3%83%88%E6%95%B0%E3%80%81%E3%83%90%E3%82%A4%E3%83%88%E3%81%82%E3%81%9F%E3%82%8A%E3%81%AE%E3%82%B9%E3%83%88%E3%83%83%E3%83%97%20%E3%83%93%E3%83%83%E3%83%88%E6%95%B0%E3%81%AA%E3%81%A9%E3%81%AE%E6%A7%8B%E6%88%90%E8%A8%AD%E5%AE%9A%E3%82%92%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%BE%E3%81%99%E3%80%82%20%E4%BB%96%E3%81%AE%20DCB%20%E3%83%A1%E3%83%B3%E3%83%90%E3%83%BC%E3%81%AF%E7%89%B9%E6%AE%8A%E6%96%87%E5%AD%97%E3%82%92%E6%8C%87%E5%AE%9A%E3%81%97%E3%80%81%E3%83%91%E3%83%AA%E3%83%86%E3%82%A3%20%E6%A4%9C%E6%9F%BB%E3%81%A8%E3%83%95%E3%83%AD%E3%83%BC%E5%88%B6%E5%BE%A1%E3%82%92%E6%9C%89%E5%8A%B9%E3%81%AB%E3%81%97%E3%81%BE%E3%81%99%E3%80%82
		// https://learn.microsoft.com/ja-jp/windows/win32/api/winbase/ns-winbase-dcb
		#pragma endregion
		DCB dcb;
		GetCommState(handle, &dcb);

		/* ----------------------------------------------------- */

		// DCBの設定を変更
		dcb.BaudRate = cfg.baudRate;
		dcb.ByteSize = (BYTE)cfg.byteSize;
		switch (cfg.parity) {
		case Config::Parity::NO:
			dcb.Parity = NOPARITY;
			break;
		case Config::Parity::EVEN:
			dcb.Parity = EVENPARITY;
			break;
		case Config::Parity::ODD:
			dcb.Parity = ODDPARITY;
			break;
		}
		dcb.fParity = (conf.parity != Config::Parity::NO);
		switch (cfg.stopBits) {
		case Config::StopBits::ONE:
			dcb.StopBits = ONESTOPBIT;
			break;
		case Config::StopBits::ONE5:
			dcb.StopBits = ONE5STOPBITS;
			break;
		case Config::StopBits::TWO:
			dcb.StopBits = TWOSTOPBITS;
			break;
		}

		/* ----------------------------------------------------- */

		// DCBをセット
		SetCommState(handle, &dcb);
	}

	bool isOpened() const {
		// デバイスがオープンしているか
		return opened;
	}

	int available(void* handle) {
		unsigned long error;
		COMSTAT stat;
		ClearCommError(handle, &error, &stat);
		return stat.cbInQue;
	}

	int read(unsigned char* data, int size) {
		// sizeバイトもしくはバッファにあるだけ受信
		unsigned long readSize;
		ReadFile(handle, data, size, &readSize, NULL);
		return readSize;
	}

	unsigned char read1byte() {
		// 1バイト受信

		unsigned char data;
		unsigned long readSize;
		int read_size = 1;

		ReadFile(handle, &data, read_size, &readSize, NULL);

		return data;
	}

	std::vector<unsigned char> read() {
		// バッファすべて受信(最低1バイト)

		unsigned long readSize;
		int read_size = available(handle);
		if (read_size == 0) {
			read_size = 1;
		}
		unsigned char* data = new unsigned char[read_size];
		ReadFile(handle, data, read_size, &readSize, NULL);
		std::vector<unsigned char> rtn;
		for (int i = 0; i < read_size; i++) {
			rtn.push_back(data[i]);
		}
		delete[] data;
		return rtn;
	}

	void clear() {
		// バッファをクリア

		PurgeComm(
			handle,
			PURGE_TXABORT |
			PURGE_RXABORT |
			PURGE_TXCLEAR |
			PURGE_RXCLEAR
		);
	}

	void clearWrite() {
		// 出力バッファをクリア
		PurgeComm(
			handle,
			PURGE_TXABORT |
			PURGE_TXCLEAR
		);
	}

	void clearRead() {
		// 入力バッファをクリア
		PurgeComm(
			handle,
			PURGE_RXABORT |
			PURGE_RXCLEAR
		);
	}

	int write(unsigned char* data, int size) {
		// 送信
		unsigned long writtenSize;
		WriteFile(
			handle,
			data,
			size,
			&writtenSize,
			NULL
		);
		return writtenSize;
	}

	int write(const std::vector<unsigned char>& data) {
		// 送信
		unsigned long writtenSize;
		int size = data.size();
		char* buff = new char[size];
		for (int i = 0; i < size; i++) {
			buff[i] = data[i];
		}

		WriteFile(handle, buff, size, &writtenSize, NULL);
		return writtenSize;
	}
};

int TLabSerialInitialize(int port) {

	// デバイス一覧
	auto list = getSerialList();

	for (const auto& info : list) {
		#pragma region cout
		// https://cpprefjp.github.io/reference/iostream/cout.html
		#pragma endregion
		std::wcout << "device name:" << info.device_name() << endl;
		std::wcout << "name:" << info.port() << "\n" << endl;
	}

	Serial serial;

	if (!serial.open(list[port], 9600)) {
		// ポート開放に失敗
		return -1;
	}

	// SerialInfo構造体にポート名とデバイス名が入っている
	SerialInfo info = serial.getInfo();
	cout << "open success" << endl;
	std::wcout << "device name:" << info.device_name() << endl;
	std::wcout << "name:" << info.port() << "\n" << endl;

	// 初期化完了
	return 0;
}

#endif