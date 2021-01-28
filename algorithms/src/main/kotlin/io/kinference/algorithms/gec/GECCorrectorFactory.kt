package io.kinference.algorithms.gec

import io.kinference.algorithms.gec.corrector.GECCorrector
import io.kinference.algorithms.gec.postprocessing.GecCorrectionPostprocessor
import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.GecCorrectionPreprocessor
import io.kinference.algorithms.gec.preprocessing.GecPreprocessor

/** Factory to create GECCorrection from GECConfig */
object GECCorrectorFactory {
    fun createGECCorrector(config: GECConfig): GECCorrector {
        val preprocessor: GecPreprocessor = GecCorrectionPreprocessor(encoder = config.encoder, useStartToken = true, evalTokenization = true)
        val postprocessor: GecPostprocessor = GecCorrectionPostprocessor()

        return GECCorrector(
            config.model,
            config.encoder,
            config.labelsVocab,
            config.dTagsVocab,
            config.verbsVocab,
            preprocessor,
            postprocessor,
            iterations = 5
        )
    }
}
