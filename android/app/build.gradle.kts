import com.android.build.api.artifact.SingleArtifact

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "ai.recall"
    compileSdk = 37

    defaultConfig {
        applicationId = "ai.recall"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Debug-signed so CI can build an installable release APK; real releases need a private key.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        noCompress += listOf("tflite", "bin")
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
    lint {
        abortOnError = true
        checkReleaseBuilds = true
        // Version-bump nags are not actionable in CI.
        disable += listOf("GradleDependency", "NewerVersionAvailable", "AndroidGradlePluginVersion", "OldTargetApi")
    }
}

kotlin {
    jvmToolchain(17)
}

// ---------------------------------------------------------------------------------------------------------------
// Model assets. models/ is git-ignored and produced by the Python side (recall/compress/export_android.py).
// If ../models/litert exists, its *.tflite / *.json / *.bin files are packaged as APK assets; otherwise the app
// builds without them and shows a "models not installed" fallback. Empty (0-byte) files are skipped.
// ---------------------------------------------------------------------------------------------------------------
abstract class CopyModelAssets : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NAME_ONLY)
    abstract val modelFiles: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        out.mkdirs()
        val copied = modelFiles.files.filter { it.isFile && it.length() > 0 }.onEach { it.copyTo(File(out, it.name), true) }
        logger.lifecycle("Recall model assets: ${if (copied.isEmpty()) "none (fallback mode)" else copied.joinToString { it.name }}")
    }
}

val modelsDir = rootProject.layout.projectDirectory.dir("../models/litert")
val copyModelAssets = tasks.register<CopyModelAssets>("copyModelAssets") {
    modelFiles.from(fileTree(modelsDir) { include("*.tflite", "*.json", "*.bin") })
    outputDir.set(layout.buildDirectory.dir("generated/recallModels"))
}

// ---------------------------------------------------------------------------------------------------------------
// Privacy policy: the merged RELEASE manifest must not request any network permission.
// Enforced twice: the verifyReleaseManifestPolicy task (runs before assembleRelease and in `check`) and the
// ManifestPolicyTest JVM unit test, which reads the same merged manifest.
// ---------------------------------------------------------------------------------------------------------------
abstract class VerifyManifestPolicy : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val manifest: RegularFileProperty

    @get:Input
    abstract val forbidden: ListProperty<String>

    @TaskAction
    fun verify() {
        val text = manifest.get().asFile.readText()
        val found = forbidden.get().filter { Regex("android:name=\"${Regex.escape(it)}\"").containsMatchIn(text) }
        if (found.isNotEmpty()) throw GradleException("Merged release manifest requests forbidden permission(s): $found")
        logger.lifecycle("Manifest policy OK: no network permission in ${manifest.get().asFile.name}")
    }
}

val forbiddenPermissions = listOf(
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.ACCESS_WIFI_STATE",
    "android.permission.CHANGE_NETWORK_STATE",
    "android.permission.CHANGE_WIFI_STATE",
    "android.permission.NEARBY_WIFI_DEVICES",
)
var releaseManifest: Provider<RegularFile>? = null

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(copyModelAssets, CopyModelAssets::outputDir)
        if (variant.buildType == "release") {
            val m = variant.artifacts.get(SingleArtifact.MERGED_MANIFEST)
            releaseManifest = m
            val verify = tasks.register<VerifyManifestPolicy>("verifyReleaseManifestPolicy") {
                manifest.set(m)
                forbidden.set(forbiddenPermissions)
            }
            tasks.matching { it.name == "assembleRelease" || it.name == "check" }.configureEach { dependsOn(verify) }
        }
    }
}

tasks.withType<Test>().configureEach {
    val m = releaseManifest
    if (m != null) {
        inputs.file(m).withPropertyName("mergedReleaseManifest").withPathSensitivity(PathSensitivity.NONE)
        systemProperty("recall.forbiddenPermissions", forbiddenPermissions.joinToString(","))
        doFirst { systemProperty("recall.mergedReleaseManifest", m.get().asFile.absolutePath) }
    }
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.core.ktx)
    implementation(libs.lifecycle.service)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.coroutines.android)
    implementation(libs.sqlcipher)
    implementation(libs.sqlite)

    testImplementation(libs.junit)
    testImplementation(libs.json)

    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.androidx.test.ext)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.uiautomator)
    androidTestImplementation(libs.compose.ui.test.junit4)
    debugImplementation(libs.compose.ui.test.manifest)
}
