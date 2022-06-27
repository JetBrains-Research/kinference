package io.kinference.utils

enum class Platform {
    JS,
    JVM
}

expect object PlatformUtils {
    val platform: Platform
}

fun <T> PlatformUtils.forPlatform(jsValue: T, jvmValue: T) = when (platform) {
    Platform.JS -> jsValue
    Platform.JVM -> jvmValue
}
