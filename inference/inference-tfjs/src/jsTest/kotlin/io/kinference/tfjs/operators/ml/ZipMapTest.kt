package io.kinference.tfjs.operators.ml

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ZipMapTest {
    private fun getTargetPath(dirName: String) = "zipmap/$dirName/"

    @Test
    fun test_zipmap_strings() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_zipmap_strings"))
    }

    @Test
    fun test_zipmap_int64s() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_zipmap_int64s"))
    }
}
