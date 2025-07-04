package io.kinference.utils

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import java.util.concurrent.CancellationException
import kotlin.test.Test
import kotlin.test.assertTrue

class ResourceDispatcherTest {
    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun testReserveCore_cancelAfterSend_triggersManualRelease() = runTest {
        val testResourcesDispatcher = TestResourcesDispatcher.createTestResourceDispatcher()

        // Pre-fill test channel to block send
        testResourcesDispatcher.testChannel.send(Unit)

        // Free up one token after a delay to simulate race
        launch {
            delay(100)
            testResourcesDispatcher.testChannel.receive() // make space
        }

        val job = launch {
            testResourcesDispatcher.reserveCore()
        }

        advanceTimeBy(50) // halfway into the race
        job.cancel()

        advanceUntilIdle()

        // Now check that the token slot is not leaked
        val success = testResourcesDispatcher.testChannel.trySend(Unit).isSuccess
        assertTrue(success, "Core should have been released after cancellation")
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun testReserveCore_cancelInGreyArea_releasesManually() = runTest {
        var leaseWasClosed = false
        val testResourcesDispatcher = TestResourcesDispatcher.createTestResourceDispatcher(
            { delay(50) },
            { leaseWasClosed = true })

        testResourcesDispatcher.testChannel.send(Unit)

        launch {
            delay(100)
            testResourcesDispatcher.testChannel.receive()
        }

        val job = launch {
            try {
                testResourcesDispatcher.reserveCore()
            } catch (_: CancellationException) {}
        }

        delay(120) // cancel in the "after lease, before return" window
        job.cancel()

        advanceUntilIdle()

        assertTrue(leaseWasClosed, "Lease should have been manually closed after cancellation")
    }
}
