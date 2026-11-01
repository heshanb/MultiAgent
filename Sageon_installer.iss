[Setup]
AppName=Sageon 1.0
AppVersion=1.0
AppVerName=Sageon 1.0
AppPublisher=Sageon Team
AppPublisherURL=https://example.com
AppSupportURL=https://example.com
AppUpdatesURL=https://example.com
VersionInfoVersion=1.0.0.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={autopf}\Sageon
DisableDirPage=no
SetupIconFile=D:\Python_project\MultiAgent\resource\sageon.ico
UninstallDisplayIcon={app}\Sageon.exe
DefaultGroupName=Sageon 1.0
OutputDir=D:\Python_project\MultiAgent\dist
OutputBaseFilename=Sageon-1.0-Windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "chinesesimplified"; MessagesFile: "D:\Python_project\MultiAgent\installer\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "D:\Python_project\MultiAgent\dist\Sageon\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Sageon"; Filename: "{app}\Sageon.exe"
Name: "{group}\{cm:UninstallProgram,Sageon}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Sageon"; Filename: "{app}\Sageon.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Sageon.exe"; Description: "{cm:LaunchProgram,Sageon}"; Flags: nowait postinstall skipifsilent