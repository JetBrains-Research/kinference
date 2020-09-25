package io.kinference.ndarray.generating

class GenerationInfo(var probs: MutableList<Float> = ArrayList(), var score: Float = -1000.0F, var wordLen: Int = 0) {
    fun add(prob: Float) {
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
