package jp.live.ugai.tugraph;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

/** The translator for Huggingface fill mask model. */
public class FillMaskBatchTranslator2 implements NoBatchifyTranslator<String[], Classifications[]> {

  private HuggingFaceTokenizer tokenizer;
  private String maskToken;
  private long maskTokenId;
  private int topK;
  private Batchifier batchifier;

  FillMaskBatchTranslator2(
      HuggingFaceTokenizer tokenizer, String maskToken, int topK, Batchifier batchifier) {
    this.tokenizer = tokenizer;
    this.maskToken = maskToken;
    this.topK = topK;
    this.batchifier = batchifier;
    Encoding encoding = tokenizer.encode(maskToken, false, false);
    maskTokenId = encoding.getIds()[0];
  }

  /** {@inheritDoc} */
  @Override
  public NDList processInput(TranslatorContext ctx, String[] inputs) throws TranslateException {
    NDManager manager = ctx.getNDManager();
    Encoding[] encodings = tokenizer.batchEncode(inputs);
    NDList[] batch = new NDList[encodings.length];
    int[] maskIndices = new int[encodings.length];
    ctx.setAttachment("maskIndices", maskIndices);
    for (int i = 0; i < encodings.length; ++i) {
      long[] indices = encodings[i].getIds();
      maskIndices[i] = FillMaskTranslator2.getMaskIndex(indices, maskToken, maskTokenId);
      batch[i] = encodings[i].toNDList(manager, false, false);
    }
    return batchifier.batchify(batch);
  }

  /** {@inheritDoc} */
  @Override
  public Classifications[] processOutput(TranslatorContext ctx, NDList list) {
    NDList[] batch = batchifier.unbatchify(list);
    int[] maskIndices = (int[]) ctx.getAttachment("maskIndices");
    Classifications[] ret = new Classifications[maskIndices.length];
    for (int i = 0; i < batch.length; ++i) {
      ret[i] = FillMaskTranslator2.toClassifications(tokenizer, batch[i], maskIndices[i], topK);
    }
    return ret;
  }
}
