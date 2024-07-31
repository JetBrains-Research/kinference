package io.kinference.utils

actual object PlatformUtils {
    actual val platform: Platform = Platform.JS

    //we don't support multithreading for JS yet
    actual val cores: Int = 1
    actual val threads: Int = 1

    // we don't limit memory for JS
    actual val maxHeap: Long = Long.MAX_VALUE
}
