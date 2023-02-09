package io.kinference.utils

enum class Platform {
    JS,
    JVM
}

/**
 * Provides data regarding current platform, such as number of actual CPU cores
 * and other platform-related properties.
 */
expect object PlatformUtils {
    val platform: Platform

    /**
     * Number of available CPU cores
     */
    val cores: Int

    /**
     * Number of threads based on the number of cores (usually cores - 1)
     */
    val threads: Int
}

fun <T> PlatformUtils.forPlatform(jsValue: T, jvmValue: T) = when (platform) {
    Platform.JS -> jsValue
    Platform.JVM -> jvmValue
}
