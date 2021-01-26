package io.kinference.algorithms.gec.tokenizer

import io.kinference.algorithms.gec.tokenizer.subword.spacy.SpacyTokenizer
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class SpacyTokenizerTest {
    private val texts = listOf(
        "Where's your bag?",
        "Horaaaaay!",
        "He is my best friend.",
        "Travelling represents energy , so if you must spend energy, it is more economic and environmentally friendly if you can translate many people with the same energy.",
        "Also, you can use better energy sources and with much better profit with public transport than the personal car (like electricity from solar or wind devices) and then your level of greenhouse gases may be reduced in an important percentage.",
        "As the Lottery would sound more appealing due to the larger winnings , greater public interest and the fact that it is easier to fill in than a pools coupon , fewer people would buy the pools coupon , and this decrease in demand would mean that pools companies such as Littlewoods and Vernon would be forced to make employees redundent due to a decrease in profits .",
        "To conclude, public transport is not useful anymore.",
        "Bus is a transportation which I rarely take.",
        "There is never a lack of work are a means for continuing employment.",
        "Eighteen year olds would not be put in jail for containing alcohol in the car either, which is another high percentage crime among teenagers.",
        "Apple is looking at buying U.K. startup for $1 billion",
        "Autonomous cars shift insurance liability toward manufacturers",
        "San Francisco considers banning sidewalk delivery robots",
        "London is a big city in the United Kingdom.",
        "Where are you?",
        "He's a very friendly man.",
        "I've been working on this for several month.",
        "What're you talking about?",
        "It's not a problem.",
        "It's Henry's problem not mine",
        "-Where's your phone?-said Gorge",
        "...Where it is, i ask you?!",
        "***Where*** it is, i ```ask``` you?!",
        "Who is the president of France?",
        "What is the capital of the United States?",
        "When was Barack Obama born?",
        "y'all should live forever!",
        "Just a simple url test: https://www.youtube.com/ and another one url https://arxiv.org/pdf/2006.03654.pdf",
        "Test for infix detection: A hard-working person.",
        "Recent progress in pre-trained neural language models has significantly improved\n the performance of many natural language processing (NLP) tasks.",
        "In this paper, we propose a new Transformer-based neural language model DeBERTa (Decoding-enhanced BERT with disentangled attention) which has been proven to be more effective than\n RoBERTa and BERT and after fine-tuning leads to better results on a wide range of NLP tasks."
    )

    val tokenizeAnswer = listOf(listOf("Where", "'s", "your", "bag", "?"),
        listOf("Horaaaaay", "!"),
        listOf("He", "is", "my", "best", "friend", "."),
        listOf("Travelling", "represents", "energy", ",", "so", "if", "you", "must", "spend", "energy", ",", "it", "is", "more", "economic", "and", "environmentally", "friendly", "if", "you", "can", "translate", "many", "people", "with", "the", "same", "energy", "."),
        listOf("Also", ",", "you", "can", "use", "better", "energy", "sources", "and", "with", "much", "better", "profit", "with", "public", "transport", "than", "the", "personal", "car", "(", "like", "electricity", "from", "solar", "or", "wind", "devices", ")", "and", "then", "your", "level", "of", "greenhouse", "gases", "may", "be", "reduced", "in", "an", "important", "percentage", "."),
        listOf("As", "the", "Lottery", "would", "sound", "more", "appealing", "due", "to", "the", "larger", "winnings", ",", "greater", "public", "interest", "and", "the", "fact", "that", "it", "is", "easier", "to", "fill", "in", "than", "a", "pools", "coupon", ",", "fewer", "people", "would", "buy", "the", "pools", "coupon", ",", "and", "this", "decrease", "in", "demand", "would", "mean", "that", "pools", "companies", "such", "as", "Littlewoods", "and", "Vernon", "would", "be", "forced", "to", "make", "employees", "redundent", "due", "to", "a", "decrease", "in", "profits", "."),
        listOf("To", "conclude", ",", "public", "transport", "is", "not", "useful", "anymore", "."),
        listOf("Bus", "is", "a", "transportation", "which", "I", "rarely", "take", "."),
        listOf("There", "is", "never", "a", "lack", "of", "work", "are", "a", "means", "for", "continuing", "employment", "."),
        listOf("Eighteen", "year", "olds", "would", "not", "be", "put", "in", "jail", "for", "containing", "alcohol", "in", "the", "car", "either", ",", "which", "is", "another", "high", "percentage", "crime", "among", "teenagers", "."),
        listOf("Apple", "is", "looking", "at", "buying", "U.K.", "startup", "for", "$", "1", "billion"),
        listOf("Autonomous", "cars", "shift", "insurance", "liability", "toward", "manufacturers"),
        listOf("San", "Francisco", "considers", "banning", "sidewalk", "delivery", "robots"),
        listOf("London", "is", "a", "big", "city", "in", "the", "United", "Kingdom", "."),
        listOf("Where", "are", "you", "?"),
        listOf("He", "'s", "a", "very", "friendly", "man", "."),
        listOf("I", "'ve", "been", "working", "on", "this", "for", "several", "month", "."),
        listOf("What", "'re", "you", "talking", "about", "?"),
        listOf("It", "'s", "not", "a", "problem", "."),
        listOf("It", "'s", "Henry", "'s", "problem", "not", "mine"),
        listOf("-Where", "'s", "your", "phone?-said", "Gorge"),
        listOf("...", "Where", "it", "is", ",", "i", "ask", "you", "?", "!"),
        listOf("*", "*", "*", "Where", "*", "*", "*", "it", "is", ",", "i", "`", "`", "`", "ask", "`", "`", "`", "you", "?", "!"),
        listOf("Who", "is", "the", "president", "of", "France", "?"),
        listOf("What", "is", "the", "capital", "of", "the", "United", "States", "?"),
        listOf("When", "was", "Barack", "Obama", "born", "?"),
        listOf("y'", "all", "should", "live", "forever", "!"),
        listOf("Just", "a", "simple", "url", "test", ":", "https://www.youtube.com/", "and", "another", "one", "url", "https://arxiv.org/pdf/2006.03654.pdf"),
        listOf("Test", "for", "infix", "detection", ":", "A", "hard", "-", "working", "person", "."),
        listOf("Recent", "progress", "in", "pre", "-", "trained", "neural", "language", "models", "has", "significantly", "improved", "\n ", "the", "performance", "of", "many", "natural", "language", "processing", "(", "NLP", ")", "tasks", "."),
        listOf("In", "this", "paper", ",", "we", "propose", "a", "new", "Transformer", "-", "based", "neural", "language", "model", "DeBERTa", "(", "Decoding", "-", "enhanced", "BERT", "with", "disentangled", "attention", ")", "which", "has", "been", "proven", "to", "be", "more", "effective", "than", "\n ", "RoBERTa", "and", "BERT", "and", "after", "fine", "-", "tuning", "leads", "to", "better", "results", "on", "a", "wide", "range", "of", "NLP", "tasks", "."),
    )

    val tokenizer = SpacyTokenizer.loadEnglish()

    @Test
    fun testSpacyTokenize(){
        for ((idx, text) in texts.withIndex()){
            val res = tokenizer.tokenize(text)
            Assertions.assertEquals(res, tokenizeAnswer[idx])
        }
    }

}
