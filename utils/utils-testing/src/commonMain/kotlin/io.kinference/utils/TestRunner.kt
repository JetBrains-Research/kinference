package io.kinference.utils

import io.kinference.utils.time.Timer
import kotlinx.coroutines.*
import kotlinx.coroutines.test.TestResult
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes

object TestRunner {
    private val logger = LoggerFactory.create("io.kinference.utils.TestRunner")

    fun runTest(platform: Platform? = null, timeout: Duration = 5.minutes, block: suspend CoroutineScope.() -> Unit): TestResult {
        if (platform != null && platform != PlatformUtils.platform) return kotlinx.coroutines.test.runTest {}

        val mark = Timer.start()
        val res = kotlinx.coroutines.test.runTest(timeout = timeout, testBody = block)
        logger.info { "[${PlatformUtils.platform}] Test took ${mark.elapsed().millis}ms" }
        return res
    }
}
