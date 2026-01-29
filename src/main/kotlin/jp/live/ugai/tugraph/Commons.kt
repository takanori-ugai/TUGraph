package jp.live.ugai.tugraph

// Hyper Parameters

/** Default learning rate used in training examples. */
const val LEARNING_RATE = 0.005f

/** Default embedding dimension for example models. */
const val DIMENSION = 20L

/** Default number of epochs for example training runs. */
const val NEPOCH = 300

/** Default batch size for example datasets. */
const val BATCH_SIZE = 1

/** Max attempts to resample a negative before falling back. */
const val NEGATIVE_RESAMPLE_CAP = 10

/** Batch size for ResultEval scoring. */
const val RESULT_EVAL_BATCH_SIZE = 8

/** L2 regularization weight for DistMult embeddings. */
const val DISTMULT_L2 = 1.0e-4f

/** L2 regularization weight for ComplEx embeddings. */
const val COMPLEX_L2 = 1.0e-4f

/** Number of negative samples per positive triple. */
const val NEGATIVE_SAMPLES = 1

/** Self-adversarial temperature for similarity-based models. */
const val SELF_ADVERSARIAL_TEMP = 0.0f

/** Number of values in a knowledge graph triple. */
const val TRIPLE = 3L

// const val NUM_ENTITIES = 202487L
// const val NUM_ENTITIES = 7658L
// const val NUM_EDGES = 37L
// const val NUM_EDGES = 57L

/** Placeholder class to keep common constants in one file. */
class Commons
