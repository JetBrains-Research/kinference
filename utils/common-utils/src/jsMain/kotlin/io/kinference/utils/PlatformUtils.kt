package io.kinference.utils

actual object PlatformUtils {
    actual val platform: Platform = Platform.JS

    //we don't support multithreading for JS yet
    actual val cores: Int = 1
    actual val availableThreads: Int = 1
}
