package io.kinference.utils

import kotlinx.coroutines.*

object TestRunner {
    fun runTest(block: suspend CoroutineScope.() -> Unit): dynamic {
        return GlobalScope.promise {
            block()
        }
    }
}

/*fun <T> TestRunner.forPlatform(jsValue: T, jvmValue: T) = when (TestRunner.platform) {
    TestRunner.Platform.JS -> jsValue
    TestRunner.Platform.JVM -> jvmValue
}*/
