package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ClipTest {
    private fun getTargetPath(dirName: String) = "clip/$dirName/"

    @Test
    fun test_clip() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip"))
    }

    @Test
    fun test_clip_default_inbounds() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds"))
    }

    @Test
    fun test_clip_default_inbounds_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_inbound() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds"))
    }

    @Test
    fun test_clip_default_int8_inbounds_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_inbounds_expanded"))
    }

    @Test
    fun test_clip_default_int8_max() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max"))
    }

    @Test
    fun test_clip_default_int8_max_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_max_expanded"))
    }

    @Test
    fun test_clip_default_int8_min() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min"))
    }

    @Test
    fun test_clip_default_int8_min_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_int8_min_expanded"))
    }

    @Test
    fun test_clip_default_max() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max"))
    }

    @Test
    fun test_clip_default_max_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_max_expanded"))
    }

    @Test
    fun test_clip_default_min() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min"))
    }

    @Test
    fun test_clip_default_min_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_default_min_expanded"))
    }

    @Test
    fun test_clip_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_example"))
    }

    @Test
    fun test_clip_example_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_example_expanded"))
    }

    @Test
    fun test_clip_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_expanded"))
    }

    @Test
    fun test_clip_inbounds() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds"))
    }

    @Test
    fun test_clip_inbounds_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_inbounds_expanded"))
    }

    @Test
    fun test_clip_outbounds() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds"))
    }

    @Test
    fun test_clip_outbounds_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_outbounds_expanded"))
    }

    @Test
    fun test_clip_splitbounds() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds"))
    }

    @Test
    fun test_clip_splitbounds_expanded() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_clip_splitbounds_expanded"))
    }
}
