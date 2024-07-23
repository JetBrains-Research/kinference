package io.kinference.utils

enum class Platform {
    JS,
    JVM
}

/**
 * Provides data regarding the current platform, such as the number of actual CPU cores
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

    /**
     * Maximum amount of heap memory that is available
     */
    val maxHeap: Long
}

fun <T> PlatformUtils.forPlatform(jsValue: T, jvmValue: T) = when (platform) {
    Platform.JS -> jsValue
    Platform.JVM -> jvmValue
}
