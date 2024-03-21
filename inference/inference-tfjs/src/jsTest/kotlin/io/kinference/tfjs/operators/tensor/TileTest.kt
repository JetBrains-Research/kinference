package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class TileTest {
    private fun getTargetPath(dirName: String) = "tile/$dirName/"

    @Test
    fun test_tile() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tile"))
    }

    @Test
    fun test_tile_precomputed() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tile_precomputed"))
    }
}
