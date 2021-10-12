package io.kinference.utils

import kotlinx.coroutines.CoroutineScope

expect object TestRunner {
    enum class Platform {
        JS,
        JVM
    }

    val platform: Platform

    fun runTest(block: suspend CoroutineScope.() -> Unit)
}

fun <T> TestRunner.forPlatform(jsValue: T, jvmValue: T) = when (TestRunner.platform) {
    TestRunner.Platform.JS -> jsValue
    TestRunner.Platform.JVM -> jvmValue
}
