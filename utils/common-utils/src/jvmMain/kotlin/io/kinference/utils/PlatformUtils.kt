package io.kinference.utils

actual object PlatformUtils {
    actual val platform: Platform = Platform.JVM

    actual val cores: Int
        get() = Runtime.getRuntime().availableProcessors()

    actual val availableThreads: Int
        get() = if (cores == 1) 1 else cores - 1
}
