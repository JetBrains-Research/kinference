package io.kinference.utils

import kotlinx.coroutines.CoroutineScope

expect object TestRunner {
    fun runTest(block: suspend CoroutineScope.() -> Unit)
}

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
