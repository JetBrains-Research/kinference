package io.kinference.utils

actual object PlatformUtils {
    actual val platform: Platform = Platform.JVM

    actual val cores: Int by lazy { Runtime.getRuntime().availableProcessors() }

    actual val threads: Int by lazy { if (cores == 1) 1 else cores - 1 }

    actual val maxHeap: Long by lazy { Runtime.getRuntime().maxMemory() }
}
