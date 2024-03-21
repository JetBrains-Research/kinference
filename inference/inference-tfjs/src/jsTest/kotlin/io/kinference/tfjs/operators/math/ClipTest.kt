package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ClipTest {
    private fun getTargetPath(dirName: String) = "clip/$dirName/"

    @Test
    fun test_clip() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip"))
    }

    @Test
    fun test_clip_default_inbounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds"))
    }

    @Test
    fun test_clip_default_inbounds_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_inbound() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds"))
    }

    @Test
    fun test_clip_default_int8_inbounds_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_max() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max"))
    }

    @Test
    fun test_clip_default_int8_max_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max_expanded"))
    }

    @Test
    fun test_clip_default_int8_min() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min"))
    }

    @Test
    fun test_clip_default_int8_min_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min_expanded"))
    }

    @Test
    fun test_clip_default_max() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max"))
    }

    @Test
    fun test_clip_default_max_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max_expanded"))
    }

    @Test
    fun test_clip_default_min() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min"))
    }

    @Test
    fun test_clip_default_min_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min_expanded"))
    }

    @Test
    fun test_clip_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_example"))
    }

    @Test
    fun test_clip_example_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_example_expanded"))
    }

    @Test
    fun test_clip_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_expanded"))
    }

    @Test
    fun test_clip_inbounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds"))
    }

    @Test
    fun test_clip_inbounds_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds_expanded"))
    }

    @Test
    fun test_clip_outbounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds"))
    }

    @Test
    fun test_clip_outbounds_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds_expanded"))
    }

    @Test
    fun test_clip_splitbounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds"))
    }

    @Test
    fun test_clip_splitbounds_expanded() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds_expanded"))
    }
}
