package io.kinference.operators.ml

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ZipMapTest {
    private fun getTargetPath(dirName: String) = "zipmap/$dirName/"

    @Test
    fun test_zipmap_strings() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_zipmap_strings"))
    }

    @Test
    fun test_zipmap_int64s() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_zipmap_int64s"))
    }
}
