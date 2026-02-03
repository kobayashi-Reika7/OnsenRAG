# Day4 フロントエンド起動スクリプト
# npm がパスにない場合、このスクリプトで Node.js をパスに追加してから起動する
$nodePath = "C:\Program Files\nodejs"
if (Test-Path $nodePath) {
    $env:Path = "$nodePath;$env:Path"
}
Set-Location $PSScriptRoot
npm run dev
