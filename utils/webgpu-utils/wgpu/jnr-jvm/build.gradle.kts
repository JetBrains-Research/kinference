import de.undercouch.gradle.tasks.download.Download

group = rootProject.group
version = rootProject.version

plugins {
    java
}

val wgpuNativeVersion = "0.11.0.1"

dependencies {
    api(project(":utils:webgpu-utils:wgpu:jnr-internal-api-jvm"))
}

val generatedSrcDir = "$buildDir/generated-src/main"
val generatedKotlinDir = "$generatedSrcDir/kotlin"
val generatedResourcesDir = "$generatedSrcDir/resources"

tasks {
    val platforms = mapOf(
        "linux" to "**.so", "macos" to "**.dylib", "windows" to "**.dll"
    )
    val wgpuReleaseDir = "$buildDir/wgpu-release"
    val wgpuHeadersDir = "$buildDir/wgpu-headers"

    val releaseFile: (platform: String) -> String = { platform -> "wgpu-$platform-x86_64-release.zip" }
    val releaseUrl: (platform: String) -> String = { platform ->
        "https://github.com/gfx-rs/wgpu-native/releases/download/v$wgpuNativeVersion/${releaseFile(platform)}"
    }

    register("downloadWgpuNative", Download::class) {
        // TODO macos-arm64 & maybe i686
        src(platforms.keys.map { releaseUrl(it) })
        dest(wgpuReleaseDir)
        onlyIfModified(true)
    }

    register("extractWgpuNativeLib", Copy::class) {
        dependsOn("downloadWgpuNative")
        platforms.forEach { (platform, fileType) ->
            from(zipTree("$wgpuReleaseDir/${releaseFile(platform)}")) {
                include(fileType)
            }
        }
        into(generatedResourcesDir)
    }

    register("extractWgpuNativeHeaders", Copy::class) {
        dependsOn("downloadWgpuNative")
        platforms.keys.first().let { platform ->
            from(zipTree("$wgpuReleaseDir/${releaseFile(platform)}")) {
                include("**.h")
            }
        }
        into(wgpuHeadersDir)
    }

    register("generateJnrMappings", JavaExec::class) {
        dependsOn("extractWgpuNativeHeaders")
        classpath = project(":utils:webgpu-utils:wgpu:jnr-generation-jvm").sourceSets["main"].runtimeClasspath
        mainClass.set("io.kinference.utils.wgpu.generation.WgpuJnrGenerator")
        args = listOf(wgpuHeadersDir, generatedKotlinDir, "io.kinference.utils.wgpu.jnr")
    }

    compileKotlin {
        dependsOn("generateJnrMappings")
    }

    processResources {
        dependsOn("extractWgpuNativeLib")
    }
}

sourceSets {
    kotlin {
        sourceSets["main"].apply {
            kotlin.srcDirs(generatedKotlinDir)
        }
    }

    main {
        resources.srcDir(generatedResourcesDir)
    }
}
