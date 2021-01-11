package io.kinference.algorithms.gec.preprocessing

import java.io.File
import java.io.InputStream

class Vocabulary {
    val token2index = HashMap<String, Int>()
    val index2token = HashMap<Int, String>()

    companion object {
        fun loadFromFile(path: String): Vocabulary {
            val vocab = Vocabulary()
            val inputStream: InputStream = File(path).inputStream()
            val lineList = ArrayList<String>()

            inputStream.bufferedReader().useLines { lines -> lines.forEach { lineList.add(it) } }
            for (line in lineList) {
                vocab.addToken(line)
            }
            return vocab
        }
    }

    fun addToken(token: String) {
        if (!token2index.containsKey(token)) {
            val index = token2index.keys.size
            token2index[token] = index
            index2token[index] = token
        }
    }

    fun addTokens(tokens: List<String>) {
        for (token in tokens) {
            addToken(token)
        }
    }

    fun getTokenIndex(token: String): Int {
        return token2index.get(token)!!
    }

    fun getTokenByIndex(idx: Int): String {
        return index2token.get(idx)!!
    }

    fun getIndexToTokenVocab(): Map<Int, String> {
        return index2token
    }

    fun getTokenToIndexVocab(): Map<String, Int> {
        return token2index
    }

    fun size(): Int {
        return token2index.size
    }

    override fun equals(other: Any?): Boolean {
        return other is Vocabulary &&
            other.index2token == index2token &&
            other.token2index == token2index
    }

    override fun hashCode(): Int {
        var result = token2index.hashCode()
        result = 31 * result + index2token.hashCode()
        return result
    }

//    @classmethod
//    def load_from_file(cls, path: Path) -> 'Vocabulary':
//    vocab = Vocabulary()
//    with path.open(mode='rt', encoding='utf-8') as f:
//    token_to_index: Dict[str, int] = json.loads(f.read())
//
//    vocab._token_to_index = token_to_index  # pylint: disable=W0212
//    vocab._index_to_token = {i: t for t, i in token_to_index.items()}  # pylint: disable=W0212
//    return vocab
}

class VerbsFormVocabulary {
    val verbs2verbs: MutableMap<String, MutableMap<String, String>> = mutableMapOf()

    companion object {
        fun setupVerbsFormVocab(path: String): VerbsFormVocabulary {
            val vocab = VerbsFormVocabulary()

            val inputStream: InputStream = File(path).inputStream()
            val lineList = ArrayList<String>()

            inputStream.bufferedReader().useLines { lines -> lines.forEach { lineList.add(it) } }
            for (line in lineList) {
                val VerbAndForm = line.split(':', limit = 2)
                val verb = VerbAndForm[0]
                val form = VerbAndForm[1]
                val inOutVerb = verb.split('_', limit = 2)
                val inVerb = inOutVerb[0]
                val outVerb = inOutVerb[1]
                vocab.addVerb(inVerb, form, outVerb)
            }
            return vocab
        }
    }

    fun addVerb(inVerb: String, form: String, outVerb: String) {
        if (verbs2verbs.containsKey(inVerb)) {
            verbs2verbs[inVerb]?.set(form, outVerb)
        } else {
            verbs2verbs[inVerb] = mutableMapOf(form to outVerb)
        }
    }
}
