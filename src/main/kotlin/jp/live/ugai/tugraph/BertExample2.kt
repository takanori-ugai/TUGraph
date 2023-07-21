package jp.live.ugai.tugraph

import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.huggingface.translator.FillMaskTranslatorFactory
import ai.djl.modality.Classifications
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ZooModel
import ai.djl.training.util.DownloadUtils
import ai.djl.training.util.ProgressBar
import ai.djl.translate.Batchifier
import ai.djl.translate.NoopTranslator
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.nio.file.Paths
import java.util.Locale
fun main() {
//    val question = "When did BBC Japan start broadcasting?"
//    val question = "Where is the capital city of japan?"
    val question = "I have watched this [MASK] and it was awesome.".lowercase().replace("[mask]", "[MASK]")
//    val resourceDocument = """
//        BBC Japan was a general entertainment Channel.
//        Which operated between December 2004 and April 2006.
//        It ceased operations after its Japanese distributor folded.
//        """.trimIndent()
    val resourceDocument = ""

    val input = QAInput(question, resourceDocument)
    val tokenizer = BertTokenizer()
    val tokenQ = tokenizer.tokenize(question.lowercase(Locale.getDefault()))
    val tokenA = tokenizer.tokenize(resourceDocument.lowercase(Locale.getDefault()))

    println("Question Token: $tokenQ")
    println("Answer Token: $tokenA")

    val token =
        tokenizer.encode(
            question.lowercase(Locale.getDefault()).replace("[mask]", "[MASK]"),
            resourceDocument.lowercase(Locale.getDefault())
        )
    println("Encoded tokens: " + token.tokens)

    println("Encoded token type: " + token.tokenTypes)
    println("Encoded token attention: " + token.attentionMask)
    println("Valid length: " + token.validLength)

    var path = Paths.get("data/bert-base-uncased-vocab.txt")
    var vocabulary = DefaultVocabulary.builder()
        .optMinFrequency(1)
        .addFromTextFile(path)
        .optUnknownToken("[UNK]")
        .build()
    val indices = token.tokens.map { token ->
        vocabulary!!.getIndex(
            token
        )
    }.toLongArray()
    println(indices.toList())
    val index = vocabulary.getIndex("[MASK]")
    val token2 = vocabulary.getToken(103)
    println("The index of the car is $index")
    println("The token of the index 2482 is $token2")

    DownloadUtils.download(
        "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/trace_bertqa.pt.gz",
        "build/pytorch/bertqa/bertqa.pt",
        ProgressBar()
    )

    val translator = BertTranslator()

    val criteria = Criteria.builder()
        .setTypes(QAInput::class.java, String::class.java)
        .optModelPath(Paths.get("build/pytorch/filled/masked_bert.pt")) // search in local folder
        .optTranslator(translator)
        .optProgress(ProgressBar()).build()

    val model: ZooModel<*, *> = criteria.loadModel()
    val pred = model.newPredictor(NoopTranslator())
//    val p0 = model.ndManager.create(longArrayOf(101, 103, 2003, 1996, 3007, 2103, 1997, 2900, 1029, 102)).reshape(10,1)
//    val p1 = model.ndManager.create(longArrayOf(1, 1,1,1,1,1,1,1,1,1)).reshape(10,1)
    val p0 = model.ndManager.create(indices).reshape(13, 1)
    val p1 = model.ndManager.create(token.attentionMask.toLongArray()).reshape(13, 1)
    val p2 = model.ndManager.create(longArrayOf(1, 1, 1, 1, 1)).reshape(5, 1)
    val res = pred.predict(NDList(p0))

    println("PREDICTION: ${res[0]}")
    println("PREDICTION: ${res[0].get(5, 0).topK(10, 0)[1]}")
    val pp = res[0].get(5, 0).topK(10, 0)[1].toLongArray().forEach {
        println(vocabulary.getToken(it))
    }

    val text = "sample questions"

    val criteria2 = Criteria.builder()
        .optApplication(Application.NLP.QUESTION_ANSWER)
        .setTypes(QAInput::class.java, String::class.java)
        .optFilter("backbone", "bert")
        .optEngine(Engine.getDefaultEngineName())
        .optProgress(ProgressBar())
        .build()

    val model2 = criteria2.loadModel()

    var predictResult: String? = null
    val question2 = "When did BBC Japan start broadcasting?"
    val resourceDocument2 = """
        BBC Japan was a general entertainment Channel.
        Which operated between December 2004 and April 2006.
        It ceased operations after its Japanese distributor folded.
    """.trimIndent()

    val input2 = QAInput(question2, resourceDocument2)

// Create a Predictor and use it to predict the output
    model2.newPredictor(BertTranslator()).use { predictor ->
        predictResult = predictor.predict(input2)
    }

    println(question2)
    println(predictResult)
    val tokenizer2 = HuggingFaceTokenizer.newInstance(Paths.get("build/pytorch/filled2/tokenizer.json"))
    val criteria3 = Criteria.builder()
        .optApplication(Application.NLP.FILL_MASK)
        .setTypes(String::class.java, Classifications::class.java)
        .optModelPath(Paths.get("build/pytorch/filled2/bert-base-uncased.pt"))
        .optTranslatorFactory(FillMaskTranslatorFactory())
        .optArgument("topK", 7)
        .optProgress(ProgressBar()).build()
    val model3 = criteria3.loadModel()
    val pred3 = model3.newPredictor()
    println(pred3.predict("[MASK] is the capital city of Japan.").topK<Classifications.Classification>(20))
}

class BertExample2

/**
 * A class that translates QAInput to String using the BERT model.
 */
class BertTranslator : Translator<QAInput, String> {
    private var tokens: MutableList<String> = mutableListOf()
    private var vocabulary: Vocabulary? = null
    private var tokenizer: BertTokenizer? = null

    /**
     * Prepare the translator for translation by initializing the vocabulary and tokenizer.
     *
     * @param ctx The translator context.
     */
    override fun prepare(ctx: TranslatorContext) {
        val path = Paths.get("data/bert-base-uncased-vocab.txt")
        vocabulary = DefaultVocabulary.builder()
            .optMinFrequency(1)
            .addFromTextFile(path)
            .optUnknownToken("[UNK]")
            .build()
        tokenizer = BertTokenizer()
    }

    /**
     * Process the input and return an NDList containing the encoded tokens.
     *
     * @param ctx The translator context.
     * @param input The input to be processed.
     * @return An NDList containing the indices, attention mask, and token types.
     */
    override fun processInput(ctx: TranslatorContext, input: QAInput): NDList {
        val token = tokenizer!!.encode(
            input.question.lowercase(Locale.getDefault()).replace("[mask]", "[MASK]"),
            input.paragraph.lowercase(Locale.getDefault())
        )
        // get the encoded tokens that would be used in precessOutput
        tokens = token.tokens
        val manager = ctx.ndManager
        // map the tokens(String) to indices(long)

        val indices = tokens.map { token ->
            vocabulary!!.getIndex(
                token
            )
        }.toLongArray()

        val attentionMask = token.attentionMask.toLongArray()
        val tokenType = token.tokenTypes.toLongArray()
        val indicesArray = manager.create(indices)
        println(indicesArray)
        val attentionMaskArray = manager.create(attentionMask)
        println(attentionMaskArray)
        val tokenTypeArray = manager.create(tokenType)
        println(tokenTypeArray)
        // The order matters
        return NDList(indicesArray, attentionMaskArray, tokenTypeArray)
    }

    /**
     * Process the output and return the predicted answer as a String.
     *
     * @param ctx The translator context.
     * @param list The NDList containing the start and end logits.
     * @return The predicted answer as a String.
     */
    override fun processOutput(ctx: TranslatorContext, list: NDList): String {
        val startLogits = list[0]
        val endLogits = list[1]
        val startIdx = startLogits.argMax().getLong().toInt()
        val endIdx = endLogits.argMax().getLong().toInt()
        return tokens.subList(startIdx, endIdx + 1).toString()
    }

    /**
     * Get the batchifier to be used.
     *
     * @return The Batchifier.STACK.
     */
    override fun getBatchifier(): Batchifier = Batchifier.STACK
}
