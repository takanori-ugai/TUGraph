package jp.live.ugai.tugraph

import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar
import org.slf4j.LoggerFactory

/**
 * An example of inference using BertQA.
 *
 *
 * See:
 *
 *
 *  * the [jupyter
 * demo](https://github.com/deepjavalibrary/djl/blob/master/jupyter/BERTQA.ipynb) with more information about BERT.
 *  * the [docs](https://github.com/deepjavalibrary/djl/blob/master/examples/docs/BERT_question_and_answer.md)
 * for information about running this example.
 *
 */
object BertQaInference {
    private val logger = LoggerFactory.getLogger(BertQaInference::class.java)

    /** Runs a simple question-answering inference demo. */
    @JvmStatic
    fun main(args: Array<String>) {
        val answer = predict()
        logger.info("Answer: {}", answer)
    }

    /**
     * Performs a QA prediction with a fixed question and paragraph.
     *
     * @return The predicted answer string.
     */
    fun predict(): String {
        val question = "When did BBC Japan start broadcasting?"
        // val paragraph = "help"
        val paragraph = (
            "BBC Japan was a general entertainment Channel. " +
                "Which operated between December 2004 and April 2006. " +
                "It ceased operations after its Japanese distributor folded."
        )
        val input = QAInput(question, paragraph)
        logger.info("Paragraph: {}", input.paragraph)
        logger.info("Question: {}", input.question)
        val criteria =
            Criteria.builder()
                .optApplication(Application.NLP.QUESTION_ANSWER)
                .setTypes(QAInput::class.java, String::class.java)
                .optFilter("backbone", "bert")
                .optEngine(Engine.getDefaultEngineName())
                .optProgress(ProgressBar())
                .build()
        criteria.loadModel().use { model ->
            model.newPredictor().use { predictor ->
                return predictor.predict(input)
            }
        }
    }
}
