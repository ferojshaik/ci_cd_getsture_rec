# Downloads gradle-wrapper.jar so you can build without opening Android Studio first.
$jarPath = "gradle\wrapper\gradle-wrapper.jar"
if (Test-Path $jarPath) {
    Write-Host "gradle-wrapper.jar already exists."
    exit 0
}
$url = "https://github.com/gradle/gradle/raw/v8.2.0/gradle/wrapper/gradle-wrapper.jar"
Write-Host "Downloading Gradle wrapper..."
New-Item -ItemType Directory -Force -Path "gradle\wrapper" | Out-Null
try {
    Invoke-WebRequest -Uri $url -OutFile $jarPath -UseBasicParsing
    Write-Host "Done. You can now run BUILD_APK.bat"
} catch {
    Write-Host "Download failed. Open the project in Android Studio and let it sync instead."
}
