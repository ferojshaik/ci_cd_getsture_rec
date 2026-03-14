import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val versionFile = file("version.properties")
fun readVersion(): Pair<Int, String> {
    if (!versionFile.exists()) {
        versionFile.writeText("versionCode=1\nversionName=1.0.0\n")
    }
    val props = Properties().apply { versionFile.reader().use { load(it) } }
    val code = (props.getProperty("versionCode", "1")).toInt()
    val name = props.getProperty("versionName", "1.0.0")
    return code to name
}
fun bumpVersion() {
    val (code, name) = readVersion()
    val parts = name.split(".")
    val major = parts.getOrElse(0) { "1" }.toIntOrNull() ?: 1
    val minor = parts.getOrElse(1) { "0" }.toIntOrNull() ?: 0
    val patch = parts.getOrElse(2) { "0" }.toIntOrNull() ?: 0
    val newName = "$major.$minor.${patch + 1}"
    versionFile.writeText("versionCode=${code + 1}\nversionName=$newName\n")
}

val (versionCodeFromFile, versionNameFromFile) = readVersion()

android {
    namespace = "com.gesture.app"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.gesture.app"
        minSdk = 24
        targetSdk = 34
        versionCode = versionCodeFromFile
        versionName = versionNameFromFile
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}
dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
}

tasks.register("incrementVersion") {
    doLast { bumpVersion() }
}
afterEvaluate {
    tasks.named("assembleDebug") { finalizedBy("incrementVersion") }
    tasks.named("assembleRelease") { finalizedBy("incrementVersion") }
}
