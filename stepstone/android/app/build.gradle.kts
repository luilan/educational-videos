import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing: the keystore and its passwords live outside the repo (default /home/codex/secrets/stepstone).
val signingProps = Properties().apply {
    val f = file(System.getenv("STEPSTONE_SIGNING") ?: "/home/codex/secrets/stepstone/signing.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "io.github.luilan.stepstone"
    compileSdk = 36
    defaultConfig {
        applicationId = "io.github.luilan.stepstone"
        minSdk = 26
        targetSdk = 36
        versionCode = 3
        versionName = "0.1.2"
        buildConfigField("String", "CATALOG_URL", "\"https://luilan.github.io/educational-videos/stepstone/catalog.json\"")
    }
    signingConfigs {
        if (signingProps.isNotEmpty()) create("release") {
            storeFile = file(signingProps.getProperty("storeFile"))
            storePassword = signingProps.getProperty("storePassword")
            keyAlias = signingProps.getProperty("keyAlias")
            keyPassword = signingProps.getProperty("keyPassword")
        }
    }
    buildTypes {
        getByName("debug") {
            // ./gradlew assembleDebug -PcatalogUrl=http://localhost:8000/catalog.json  (with adb reverse tcp:8000 tcp:8000)
            (project.findProperty("catalogUrl") as String?)?.let { buildConfigField("String", "CATALOG_URL", "\"$it\"") }
        }
        getByName("release") {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }
    buildFeatures { compose = true; buildConfig = true }
    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.systemProperty("robolectric.pixelCopyRenderMode", "hardware") }   // lets UI tests take screenshots
    }
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.04.01"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
    testImplementation("org.robolectric:robolectric:4.16.1")
    testImplementation("androidx.test.ext:junit:1.3.0")
    testImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
