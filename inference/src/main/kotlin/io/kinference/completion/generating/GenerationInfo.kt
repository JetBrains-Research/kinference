package io.kinference.completion.generating

class GenerationInfo(initProbs: List<Double> = ArrayList(), var score: Double = -1000.0, var wordLen: Int = 0) {
    var probs: MutableList<Double>

    init {
        probs = initProbs.toMutableList()
    }

    fun add(prob: Double) {
        probs.add(prob)
    }

    fun trim(left: Int, right: Int? = null): GenerationInfo {
        if (right == null) {
            val realRight = left
            val realLeft = 0
            probs = probs.subList(realLeft, realRight)
        } else {
            probs = probs.subList(left, right)
        }
        return this
    }

//    fun probs(): List<Float> {
//        return probs
//    }

//    fun toDict() {
//        return {"probs": self._probs, "word_len": self.word_len}
//    }
//
//    @staticmethod
//    def from_dict(dict):
//    return GenerationInfo(dict["probs"], dict["word_len"])
}
