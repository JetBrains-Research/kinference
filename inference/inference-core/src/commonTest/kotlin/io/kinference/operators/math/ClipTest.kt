package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ClipTest {
    private fun getTargetPath(dirName: String) = "clip/$dirName/"

    @Test
    fun test_clip() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip"))
    }

    @Test
    fun test_clip_default_inbounds() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds"))
    }

    @Test
    fun test_clip_default_inbounds_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_inbound() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds"))
    }

    @Test
    fun test_clip_default_int8_inbounds_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_max() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max"))
    }

    @Test
    fun test_clip_default_int8_max_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max_expanded"))
    }

    @Test
    fun test_clip_default_int8_min() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min"))
    }

    @Test
    fun test_clip_default_int8_min_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min_expanded"))
    }

    @Test
    fun test_clip_default_max() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max"))
    }

    @Test
    fun test_clip_default_max_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max_expanded"))
    }

    @Test
    fun test_clip_default_min() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min"))
    }

    @Test
    fun test_clip_default_min_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min_expanded"))
    }

    @Test
    fun test_clip_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_example"))
    }

    @Test
    fun test_clip_example_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_example_expanded"))
    }

    @Test
    fun test_clip_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_expanded"))
    }

    @Test
    fun test_clip_inbounds() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds"))
    }

    @Test
    fun test_clip_inbounds_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds_expanded"))
    }

    @Test
    fun test_clip_outbounds() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds"))
    }

    @Test
    fun test_clip_outbounds_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds_expanded"))
    }

    @Test
    fun test_clip_splitbounds() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds"))
    }

    @Test
    fun test_clip_splitbounds_expanded() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds_expanded"))
    }
}
