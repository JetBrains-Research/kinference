package io.kinference.operators.ml

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ZipMapTest {
    private fun getTargetPath(dirName: String) = "zipmap/$dirName/"

    @Test
    fun test_zipmap_strings() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_zipmap_strings"))
    }

    @Test
    fun test_zipmap_int64s() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_zipmap_int64s"))
    }
}
