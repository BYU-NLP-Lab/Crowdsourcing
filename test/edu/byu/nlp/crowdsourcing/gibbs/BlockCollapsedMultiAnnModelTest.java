/**
 * Copyright 2013 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing.gibbs;

import static java.lang.Math.log;
import static org.apache.commons.math3.special.Gamma.logGamma;
import static org.fest.assertions.Assertions.assertThat;

import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.fest.assertions.Assertions;
import org.fest.assertions.Delta;
import org.junit.Test;

import edu.byu.nlp.crowdsourcing.ModelInitialization;
import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.TestUtil;
import edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModel.ModelBuilder;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.JsonDatasetMocker;
import edu.byu.nlp.dataset.BasicSparseFeatureVector;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;
import edu.byu.nlp.util.asserts.MoreAsserts;

/**
 * @author rah67
 *
 */
public class BlockCollapsedMultiAnnModelTest {

  private static double B_THETA = 1.0 / 3.0;
  private static double B_MU = 0.75;
  private static double C_MU = 2;
  private static double B_GAMMA = 0.75;
  private static int NUM_ANNOTATORS = 2;
  private static double C_GAMMA = 2.0;
  private static double B_PHI = 1.23;

  @Test
  public void testComputeYSum() {
    double[] countOfJYAndA = new double[] {0, 10, 1};
    double numAnnsPerJAndY = DoubleArrays.sum(countOfJYAndA);
    int[] a_ij = new int[]{0, 3, 1};
    int docJCount = (int) IntArrays.sum(a_ij);
    
    double actual = BlockCollapsedMultiAnnModelMath.computeYSum(
        countOfJYAndA, numAnnsPerJAndY, a_ij, docJCount);
    double expected =
        GammaFunctions.logRisingFactorial(0, 0) +
        GammaFunctions.logRisingFactorial(10, 3) +
        GammaFunctions.logRisingFactorial(1, 1) -
        GammaFunctions.logRisingFactorial(11, 4);
    
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }

  @Test
  public void testComputeYMSum() {
    double logCountOfYAndM = 3;
    double lambda = 1;

    double actual = BlockCollapsedMultiAnnModelMath.computeYMSum(logCountOfYAndM, lambda);
    double expected = logCountOfYAndM; // there should be no change since lambda is 1
    
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }
  
  @Test
  public void testComputeMSum() {
    final int[] indices = new int[] { 1, 5 };
    final double[] values = new double[] { 2.0, 3.0 };
    SparseFeatureVector instance = new BasicSparseFeatureVector(indices, values);
    int docSize = (int) Math.round(instance.sum());
    double[] countOfMAndX = new double[]{ 10, 9, 8, 7, 6, 5, 6 };
    double numFeaturesPerM = DoubleArrays.sum(countOfMAndX);
    
    double actual = 
        BlockCollapsedMultiAnnModelMath.computeMSum(instance, docSize, countOfMAndX, numFeaturesPerM);
    double expected = GammaFunctions.logRisingFactorial(countOfMAndX[indices[0]], (int) values[0]) +
        GammaFunctions.logRisingFactorial(countOfMAndX[indices[1]], (int) values[1]) -
        GammaFunctions.logRisingFactorial(numFeaturesPerM, docSize);
    
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }

  @Test
  public void testComputeMSums() {

    ModelBuilder builder = newStubBuilder();
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    SparseFeatureVector instance = builder.getData().iterator().next().asFeatureVector();
    double docsize = model.docSize()[0];
    int numLabels = builder.getData().getInfo().getNumClasses();
    // don't bother removing an instance here, since both calculations should be off in the same way
    
    double[] expected = new double[numLabels];
    for (int d=0; d<numLabels; d++){
      double msum = BlockCollapsedMultiAnnModelMath.computeMSum(instance, docsize, model.getCountOfMAndX()[d], model.getNumFeaturesPerM()[d]);
      expected[d] = msum;
    }
    
    double lambda = 1; 
    double[] actual = BlockCollapsedMultiAnnModelMath.computeMSums(instance, docsize, 
        model.getCountOfMAndX(), model.getNumFeaturesPerM(), builder.getData().getInfo().getNumClasses(), lambda);

    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }

  @Test
  public void testComputeYSums() {

    ModelBuilder builder = newStubBuilder();
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    int docIndex = 0;
    // don't bother removing an instance here, since both calculations should be off in the same way
    double[] expected = model.getLogCountOfY().clone();
    for (int c = 0; c < expected.length; c++) {
      for (int j = 0; j < model.numAnnotators(); j++) {
        expected[c] +=
            BlockCollapsedMultiAnnModelMath.computeYSum(model.getCountOfJYAndA()[j][c], model.getNumAnnsPerJAndY()[j][c], model.getA()[docIndex][j],
                model.docJCount()[docIndex][j]);
      }
    }
    
    double lambda = 1; 
    double[] actual = BlockCollapsedMultiAnnModelMath.computeYSums(
        docIndex, null, model.getLogCountOfY(), model.getCountOfJYAndA(), model.getNumAnnsPerJAndY(), 
        model.getA(), model.docJCount(), model.numAnnotators(), lambda);

    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }
  
  @Test
  public void testLogJoint() {
    ModelBuilder builder = newStubBuilder();
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    
    double gamma0Diag = B_GAMMA * C_GAMMA;
    double gamma0Off = (1.0 - B_GAMMA) / 2 * C_GAMMA;
    double gamma1Diag = B_GAMMA * C_GAMMA;
    double gamma1Off = (1.0 - B_GAMMA) / 2 * C_GAMMA;
    double muDiag = B_MU * C_MU;
    double muOff = (1.0 - B_MU) / 2.0 * C_MU;

    double expected = 
        /* theta */
        logGamma(B_THETA + 2) + logGamma(B_THETA + 2) + logGamma(B_THETA + 1) +
        
        /* mu */
        logGamma(muDiag + 2) + logGamma(muOff) + logGamma(muOff) -      /* k = 0 */
        logGamma(muDiag + 2 + muOff + muOff) +
        logGamma(muOff + 1) + logGamma(muDiag) + logGamma(muOff + 1) -  /* k = 1 */
        logGamma(muOff + 1 + muDiag + muOff + 1) +
        logGamma(muOff) + logGamma(muOff + 1) + logGamma(muDiag) -      /* k = 2 */
        logGamma(muOff + muOff + 1 + muDiag) +

        /* phi */
        logGamma(B_PHI + 1) + logGamma(B_PHI + 3) + logGamma(B_PHI + 5) + logGamma(B_PHI + 1) + /* k = 0 */
        logGamma(B_PHI + 2) - logGamma(5 * B_PHI + 1 + 3 + 5 + 1 + 2) +
        logGamma(B_PHI) + logGamma(B_PHI + 1) + logGamma(B_PHI) + logGamma(B_PHI + 4) + /* k = 1 */
        logGamma(B_PHI) - logGamma(5 * B_PHI + 1 + 4) +
        logGamma(B_PHI) + logGamma(B_PHI + 1) + logGamma(B_PHI) + logGamma(B_PHI) + /* k = 2 */
        logGamma(B_PHI + 1) - logGamma(5 * B_PHI + 1 + 1) +

        /* gamma */
        logGamma(gamma0Diag + 1) + logGamma(gamma0Off) + logGamma(gamma0Off) -      /* j = 0, k = 0 */  
        logGamma(gamma0Diag + 1 + gamma0Off + gamma0Off) +
        logGamma(gamma0Off + 2) + logGamma(gamma0Diag) + logGamma(gamma0Off) -      /* j = 0, k = 1 */ 
        logGamma(gamma0Off + 2 + gamma0Diag + gamma0Off) +
        logGamma(gamma0Off) + logGamma(gamma0Off) + logGamma(gamma0Diag + 1) -      /* j = 0, k = 2 */
        logGamma(gamma0Off + gamma0Off + gamma0Diag + 1) +
        logGamma(gamma1Diag) + logGamma(gamma1Off) + logGamma(gamma1Off) -          /* j = 1, k = 0 */
        logGamma(gamma1Diag + gamma1Off + gamma1Off) +
        logGamma(gamma1Off + 1) + logGamma(gamma1Diag + 2) + logGamma(gamma1Off) -  /* j = 1, k = 1 */ 
        logGamma(gamma1Off + 1 + gamma1Diag + 2 + gamma1Off) +
        logGamma(gamma1Off) + logGamma(gamma1Off + 1) + logGamma(gamma1Diag) -      /* j = 1, k = 2 */ 
        logGamma(gamma1Off + gamma1Off + 1 + gamma1Diag);

    double actual = model.logJoint();
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }
  
  /**
   * Test that the "efficient" computeCoefficients is correct, as judged by computing the full
   * joint. note(pfelt): this is a crucial test for model correctness. 
   * If it is not passing, something is wrong.
   * Similarly, if you update the model in some way that breaks this test, repair it  
   * again very carefully.
   */
  @Test
  public void testComputeCoefficientsAgainstJoint() {
    for (double lambda=0; lambda<=1; lambda+= 0.1){
      double[] docWeights = new double[]{1,1,1,1,lambda};
      for (int docIndex=0; docIndex<newStubBuilder().getData().getInfo().getNumDocuments(); docIndex++){
        testComputeCoefficientsAgainstJoint( docIndex, docWeights);
      }
    }
  }
  private void testComputeCoefficientsAgainstJoint(int docIndex, double[] docWeights) {
    ModelBuilder builder = newStubBuilder(docWeights);
    // No need to increment, decrement, because we're not testing that part of the equation (i.e.
    // They are the same in both methods of computing the coefficients.
    int numLabels = builder.getData().getInfo().getNumClasses();
    int numAnnotators = builder.getPriors().getNumAnnotators();
    
    double[] expected = new double[numLabels * numLabels];
    int coeffIndex = 0;
    for (int c = 0; c < numLabels; c++) {
      // Need to update the summary stats :(
      builder.getY()[docIndex] = c;
      for (int d = 0; d < numLabels; d++) {
        builder.getM()[docIndex] = d;
        BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
        expected[coeffIndex++] = model.logJoint();
      }
    }
    
    builder = newStubBuilder(docWeights);
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    // get the right instance
    Iterator<DatasetInstance> itr = builder.getData().iterator();
    for (int i=0; i<docIndex; i++){
      itr.next();
    }
    SparseFeatureVector instance = itr.next().asFeatureVector();
    
    // Above, we computed the joint as if we had already removed the document; so we do so here as
    // well. That is, we remove a document and then pass it back in to be added back in the math.
    model.decrementCounts(docIndex, instance, docWeights[docIndex]);
    double[] actual = model.computeCoefficients(docIndex, instance, numAnnotators, numLabels);
    
    DoubleArrays.logNormalizeToSelf(expected);
    DoubleArrays.logNormalizeToSelf(actual);
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }
  
  @Test
  public void testComputeCoefficients() {
    double[] ySums = new double[] { -1.2, -3.4, -2.1 };
    double[] mSums = new double[] { -3.3, -2.2, -1.1 };
    double[][] ymSums = new double[][] {
        { log(15), log(12), log(24) },
        { log(13), log(52), log(32) },
        { log(1), log(23), log(36) }
    };
    
    double[] expected = new double[] {
        ySums[0] + mSums[0] + ymSums[0][0],
        ySums[0] + mSums[1] + ymSums[0][1],
        ySums[0] + mSums[2] + ymSums[0][2],
        ySums[1] + mSums[0] + ymSums[1][0],
        ySums[1] + mSums[1] + ymSums[1][1],
        ySums[1] + mSums[2] + ymSums[1][2],
        ySums[2] + mSums[0] + ymSums[2][0],
        ySums[2] + mSums[1] + ymSums[2][1],
        ySums[2] + mSums[2] + ymSums[2][2],
    };

    double[] actual = BlockCollapsedMultiAnnModel.computeCoefficients(
        ySums, mSums, ymSums, null, -1);
    
    assertThat(actual).isEqualTo(expected, Delta.delta(1e-10));
  }
  
  @Test
  public void testJSONData() throws FileNotFoundException{
    
    Dataset data = JsonDatasetMocker.buildTestDatasetFromJson(JsonDatasetMocker.jsonInstances(System.currentTimeMillis()));
    RandomGenerator rnd = new MersenneTwister(1);
    
    ModelBuilder builder = (ModelBuilder) new BlockCollapsedMultiAnnModel.ModelBuilder()
      .setPriors(new PriorSpecification(B_THETA, B_MU, C_MU, .75, C_GAMMA, B_PHI, -1, false, 3))
      .setData(data)
      .setYInitializer(new ModelInitialization.BaselineInitializer(rnd))
      .setMInitializer(new ModelInitialization.BaselineInitializer(rnd))
      .setRnd(rnd);
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel) builder.build();

    // make sure annotations are in the right order 
    int[][][] a = model.getA();
    for (int i=0; i<a.length; i++){
      for (int j=0; j<a[i].length; j++){
        for (int k=0; k<a[i][j].length; k++){
          int val = a[i][j][k];
          Assertions.assertThat(
              (i==0 && j==0 && k==0 && val==1) ||
              (i==0 && j==1 && k==1 && val==1) ||
              (i==3 && j==0 && k==0 && val==1) ||
              (i==3 && j==0 && k==1 && val==1) ||
              (i==4 && j==2 && k==1 && val==1) ||
              (i==6 && j==0 && k==0 && val==1) ||
              (i==7 && j==0 && k==0 && val==1) ||
              val==0
              );
        }
      }
    }
    
    // make sure data map is correct
    int i=0;
    Map<String,Integer> indices = model.getInstanceIndices();
    for (DatasetInstance inst: data){
      Assertions.assertThat(indices.get(inst.getInfo().getSource())).isEqualTo(i++);
    }
    i=0;
    for (DatasetInstance inst: model.data){
      Assertions.assertThat(indices.get(inst.getInfo().getSource())).isEqualTo(i++);
    }

  }

  private ModelBuilder newStubBuilder() {
    return newStubBuilder(null);
  }
  private ModelBuilder newStubBuilder(double[] lambda) {
    Dataset stubData = TestUtil.stubDataset();
    PriorSpecification stubPriors =
        new PriorSpecification(B_THETA, B_MU, C_MU, B_GAMMA, C_GAMMA, B_PHI, -1, false, NUM_ANNOTATORS);
    int y[] = new int[] {0, 1, 2, 1, 0};
    int m[] = new int[] {0, 0, 1, 2, 0};
    RandomGenerator rnd = new MersenneTwister(1);
    AssignmentInitializer nullInitializer = new AssignmentInitializer() {
      @Override
      public void setData(Dataset data,Map<String,Integer> instanceIndices) { /* ignore */ }
      @Override
      public void initialize(int[] assignments) { /* ignore */ }
    };
    return (ModelBuilder) new BlockCollapsedMultiAnnModel.ModelBuilder()
      .setPriors(stubPriors)
      .setData(stubData)
      .setDocumentWeights(lambda)
      .setY(y)
      .setYInitializer(nullInitializer)
      .setM(m)
      .setMInitializer(nullInitializer)
      .setRnd(rnd);
  }
  
  /**
   * Test method for {@link edu.byu.nlp.crowdsourcing.gibbs.BlockCollapsedMultiAnnModel#newBuilder(edu.byu.nlp.crowdsourcing.PriorSpecification, edu.byu.nlp.cluster.Dataset, int, edu.byu.nlp.crowdsourcing.BlockCollapsedMultiAnnModel.AccuracyInitializer, edu.byu.nlp.crowdsourcing.BlockCollapsedMultiAnnModel.AccuracyInitializer, edu.byu.nlp.crowdsourcing.BlockCollapsedMultiAnnModel.AssignmentInitializer, org.apache.commons.math3.random.RandomGenerator)}.
   */
  @Test
  public void testNewBuilder() {
    ModelBuilder modelBuilder = newStubBuilder();
    // Assume builder hasn't modified y and m. We need these to verify that they don't changed
    // when build() is invoked.
    int y[] = modelBuilder.getY().clone();
    int m[] = modelBuilder.getM().clone();
    
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel) modelBuilder.build();
    
    // Ensure that "build" did not modify y and m
    assertThat(model.getY()).isEqualTo(y);
    assertThat(model.getM()).isEqualTo(m);
    
    double[] expectedLogCountOfY = new double[] { log(B_THETA + 2), log(B_THETA + 2), log(B_THETA + 1) };
    assertThat(model.getLogCountOfY()).isEqualTo(expectedLogCountOfY, Delta.delta(1e-14));
    
    double[][] expectedCountOfMAndX = new double[][] {
        { B_PHI + 1, B_PHI + 3, B_PHI + 5, B_PHI + 1, B_PHI + 2 },
        { B_PHI, B_PHI + 1, B_PHI, B_PHI + 4, B_PHI },
        { B_PHI, B_PHI + 1, B_PHI, B_PHI, B_PHI + 1 }};
    MoreAsserts.assertMatrixEquals(model.getCountOfMAndX(), expectedCountOfMAndX, 1e-14);
    
    double[] numFeaturesPerM = new double[] {
        B_PHI + 1 + B_PHI + 3 + B_PHI + 5 + B_PHI + 1 + B_PHI + 2,
        B_PHI + B_PHI + 1 + B_PHI + B_PHI + 4 + B_PHI,
        B_PHI + B_PHI + 1 + B_PHI + B_PHI + B_PHI + 1,
    };
    assertThat(model.getNumFeaturesPerM()).isEqualTo(numFeaturesPerM, Delta.delta(1e-14));
    
    // The prior counts on the diagonal and off the diagonal. Counts are obtained by
    // Multiplying accuracies by the concentration parameter.
    double diag = B_MU * C_MU;
    // We assume probability of incorrect classes is uniformly spread given an
    // priory accuracy B_MU. To obtain counts, we multiply be C_MU.
    double offDiag = ((1.0 - B_MU) / (3 /* num_classes) */ - 1)) * C_MU;
    double[][] expectedLogCountOfYAndM = new double[][] {
        { log(diag + 2), log(offDiag), log(offDiag) },
        { log(offDiag + 1), log(diag), log(offDiag + 1) },
        { log(offDiag), log(offDiag + 1), log(diag) }};
    MoreAsserts.assertMatrixEquals(model.getLogCountOfYAndM(), expectedLogCountOfYAndM, 1e-14);
    
    double[] expectedLogSumCountOfYAndM = new double[] {
        log(diag + 2 + offDiag + offDiag),
        log(offDiag + 1 + diag + offDiag + 1),
        log(offDiag + offDiag + 1 + diag)
      };
    assertThat(model.getLogSumCountOfYAndM()).isEqualTo(expectedLogSumCountOfYAndM,
                                                        Delta.delta(1e-14));
    // See above.
    double diag0 = B_GAMMA * C_GAMMA;
    double offDiag0 = ((1.0 - B_GAMMA) / (3 /* num_classes) */ - 1)) * C_GAMMA;
    double diag1 = B_GAMMA * C_GAMMA;
    double offDiag1 = ((1.0 - B_GAMMA) / (3 /* num_classes) */ - 1)) * C_GAMMA;
    double[][][] expectedCountOfJYAndA = new double[][][] {
        /* Annotator 0 */
        {{diag0 + 1, offDiag0, offDiag0},
         {offDiag0 + 2, diag0, offDiag0},
         {offDiag0, offDiag0, diag0 + 1}},
         
        /* Annotator 1 */
        {{diag1, offDiag1, offDiag1},
         {offDiag1 + 1, diag1 + 2, offDiag1},
         {offDiag1, offDiag1 + 1, diag1}}
    };
    MoreAsserts.assertMatrixEquals(model.getCountOfJYAndA()[0], expectedCountOfJYAndA[0], 1e-14);
    MoreAsserts.assertMatrixEquals(model.getCountOfJYAndA()[1], expectedCountOfJYAndA[1], 1e-14);

    double[][] expectedNumAnnsPerJAndY = new double[][] {
        { diag0 + 1 + offDiag0 + offDiag0,
          offDiag0 + 2 + diag0 + offDiag0,
          offDiag0 + offDiag0 + diag0 + 1 },
        { diag1 + offDiag1 + offDiag1,
          offDiag1 + 1 + diag1 + 2 + offDiag1,
          offDiag1 + offDiag1 + 1 + diag1 },
    };
    MoreAsserts.assertMatrixEquals(model.getNumAnnsPerJAndY(), expectedNumAnnsPerJAndY, 1e-14);
    
    int[][][] expectedA = new int[][][] {
        /* Doc 0 */
        {{ 1, 0, 0},
         { 0, 0, 0}},
        /* Doc 1 */
        {{ 0, 0, 0},
         { 0, 1, 0}},
        /* Doc 2 */
        {{ 0, 0, 1},
         { 0, 1, 0}},
        /* Doc 3 */
        {{ 2, 0, 0},
         { 1, 1, 0}},
    };
    MoreAsserts.assertMatrixEquals(model.getA()[0], expectedA[0]);
    MoreAsserts.assertMatrixEquals(model.getA()[1], expectedA[1]);
    MoreAsserts.assertMatrixEquals(model.getA()[2], expectedA[2]);
    MoreAsserts.assertMatrixEquals(model.getA()[3], expectedA[3]);
  }


// note(pfelt): the following label switching tests are interesting, 
// but they don't pass and I'm not sure that they even should. That is, 
// the log probability of two settings will be *mostly* the same, but 
// not quite because there are annotations involved, and diagonal 
// gammas will match diagonal prior settings better than off-diagonal 
// gammas. I'm leaving them here for future thought, but I don't 
// think they'll work as is. The diagonalization algorithms are tested 
// elsewhere, independent of model logic.
//  @Test
//  public void testFixLabelSwitching() {
//    // TODO(rhaertel): use builder to generate desired properties
//    ModelBuilder builder = newStubBuilder();
//    // necessary for state to be able to generate samples when 
//    // methods like getMu() are called (by fixlabelswitching).
//    // does not affect computation.
//    builder.setRnd(new JDKRandomGenerator()); 
//    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
//    
//    // Manually ensure that first gamma is label-switched and second gamma is distinctive.
//    // This is possible because getCountOfJYandA() returns a direct, mutable reference to the
//    // internal state of the model
//    double[][][] countOfJYAndA = model.getCountOfJYAndA();
//    countOfJYAndA[0][0] = new double[]{0.1, 0.8, 0.1};
//    countOfJYAndA[0][1] = new double[]{0.1, 0.1, 0.8};
//    countOfJYAndA[0][2] = new double[]{0.8, 0.1, 0.1};
//    
//    countOfJYAndA[1][0] = new double[]{0.1, 0.1, 0.8};
//    countOfJYAndA[1][1] = new double[]{0.8, 0.1, 0.1};
//    countOfJYAndA[1][2] = new double[]{0.1, 0.8, 0.1};
//    
//    // Manually ensure that a 3x3 mu is label-switched. As before, getLogCountOfYAndM()
//    // returns a direct, mutable reference to the internal state of the model.
//    double[][] countOfYAndM = model.getLogCountOfYAndM();
//    countOfYAndM[0] = new double[]{0.1, 0.8, 0.1};
//    countOfYAndM[1] = new double[]{0.1, 0.1, 0.8};
//    countOfYAndM[2] = new double[]{0.8, 0.1, 0.1};
//    
//    // The joint distribution should not change as a result of permutating any labels.
//    // This particular method touches all of the sufficient statistics and cached values
//    // (but does not touch individual assignments and values cached from the dataset).
//    double logJointBeforeCall = model.logJoint();
//    MultiAnnState state = model.getCurrentState();
//    int[] gold = Datasets.labels(builder.getData());
////    state = BlockCollapsedMultiAnnModelMath.fixLabelSwitching(state, gold, DiagonalizationMethod.GOLD, -1, true, model.data);
//    builder.setY(state.getY());
//    builder.setM(state.getM());
//    model = (BlockCollapsedMultiAnnModel)builder.build();
//    
////    MultiAnnState state = model.getCurrentState();
////    state = BlockCollapsedMultiAnnModelMath.fixLabelSwitching(state, gold, diagonalizationMethod, goldInstancesForDiagonalization, diagonalizationWithFullConfusionMatrix, data);
//    assertThat(logJointBeforeCall).isEqualTo(model.logJoint(), Delta.delta(1e-10));
//
//    // CountOfJYAndA
//    MoreAsserts.assertMatrixEquals(model.getCountOfJYAndA()[0], 
//        new double[][]{
//          {0.8,0.1,0.1},
//          {0.1,0.8,0.1},
//          {0.1,0.1,0.8}}, 
//        1e-20);
//    MoreAsserts.assertMatrixEquals(model.getCountOfJYAndA()[1], 
//        new double[][]{
//          {0.1,0.8,0.1},
//          {0.1,0.1,0.8},
//          {0.8,0.1,0.1}}, 
//        1e-20);
//    
//    // LogCountOfYAndM
//    MoreAsserts.assertMatrixEquals(model.getLogCountOfYAndM(), 
//        new double[][]{
//          {0.8, 0.1, 0.1},
//          {0.1, 0.8, 0.1},
//          {0.1, 0.1, 0.8}}, 
//        1e-20);
//    
//    // CountOfMAndX
//    MoreAsserts.assertMatrixEquals(model.getCountOfMAndX(), 
//        new double[][]{
//          {2.23, 4.23, 2.23, 1.23, 3.23},
//          {1.23, 2.23, 1.23, 5.23, 1.23},
//          {1.23, 2.23, 1.23, 1.23, 2.23},
//          }, 
//        1e-20);
//
//    // M 
//    assertThat(model.getM()).isEqualTo(new int[]{0, 0, 1, 2}); 
//    
//    // numFeaturesPerM
//    assertThat(model.getNumFeaturesPerM())
//        .isEqualTo(new double[]{13.15, 11.15, 8.15}, Delta.delta(1e-10)); 
//  }
//  
//  @Test
//  public void testFixLabelSwitching2(){
//    // TODO(rhaertel): use builder to generate desired properties
//    ModelBuilder builder = newStubBuilder();
//    // manually set y's and m's with interesting counts
//    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
//    model.getM()[0] = 0;
//    model.getM()[0] = 1;
//    model.getM()[0] = 1;
//    model.getM()[0] = 1;
//
//    model.getY()[0] = 0;
//    model.getY()[0] = 1;
//    model.getY()[0] = 1;
//    model.getY()[0] = 1;
//
//    // manually ensure that gamma is label-switched
//    double[][][] countOfJYAndA = model.getCountOfJYAndA();
//    countOfJYAndA[0][0] = new double[]{0, 1, 0};
//    countOfJYAndA[0][1] = new double[]{0, 0, 1};
//    countOfJYAndA[0][2] = new double[]{1, 0, 0};
//    
//    // manually ensure that a 3x3 mu is label-switched
//    double[][] countOfYAndM = model.getLogCountOfYAndM();
//    countOfYAndM[0] = new double[]{0, 1, 0};
//    countOfYAndM[1] = new double[]{0, 0, 1};
//    countOfYAndM[2] = new double[]{1, 0, 0};
//    
//    // count the y's and m's
//    int[] yCounts = new int[model.getY().length];
//    for (int y: model.getY()){
//      ++yCounts[y];
//    }
//    Arrays.sort(yCounts);
//    int[] mCounts = new int[model.getM().length];
//    for (int m: model.getM()){
//      ++mCounts[m];
//    }
//    Arrays.sort(mCounts);
//    
//    // Do transformation
//    model.fixLabelSwitching(false);
//
//    // make sure y and m counts haven't changed
//    // their labels may have changed, but the 
//    // most common label should be just as common now as before, etc.
//    int[] yCounts2 = new int[model.getY().length];
//    for (int y: model.getY()){
//      ++yCounts2[y];
//    }
//    Arrays.sort(yCounts2);
//    int[] mCounts2 = new int[model.getM().length];
//    for (int m: model.getM()){
//      ++mCounts2[m];
//    }
//    Arrays.sort(mCounts2);
//
//    assertThat(yCounts).isEqualTo(yCounts2);
//    assertThat(mCounts).isEqualTo(mCounts2);
//  }
  
  @Test
  public void testMeanSquaredDistanceFrom01(){
    double[][] mat = new double[][]{
        {1,0,0},
        {0,1,0},
        {0,0,1}
    };
    double score = BlockCollapsedMultiAnnModelMath.getMeanSquaredDistanceFrom01(mat);
    assertThat(score).isEqualTo(0);

    mat = new double[][]{
        {1,.5,0},
        {0,1,.2},
        {.1,0,1}
    };
    score = BlockCollapsedMultiAnnModelMath.getMeanSquaredDistanceFrom01(mat);
    assertThat(score).isEqualTo((.5*.5)+(.2*.2)+(.1*.1));
  }
  
  @Test
  public void testIntArrayCounts(){
	  IntArrayCounter counts = new IntArrayCounter(5,5);
	  counts.increment(new int[]{1,5,2,3,1});
	  counts.increment(new int[]{1,5,2,3,1});
	  counts.increment(new int[]{2,1,5,2,4});
	  assertThat(counts.argmax(0)).isEqualTo(1);
	  assertThat(counts.argmax(1)).isEqualTo(5);
	  assertThat(counts.argmax(2)).isEqualTo(2);
	  assertThat(counts.argmax(3)).isEqualTo(3);
	  assertThat(counts.argmax(4)).isEqualTo(1);
	  
  }
  
  
  @Test
  public void testDecrementCounts(){
    for (double lambda=0; lambda<=1; lambda+=0.1){
      testDecrementCounts(lambda);
    }
  }
  private void testDecrementCounts(double lambda){

    ModelBuilder builder = newStubBuilder();
    int docIndex = 0; 
    int numLabels = builder.getData().getInfo().getNumClasses();
    int numAnnotators = builder.getPriors().getNumAnnotators();
    
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    SparseFeatureVector instance = builder.getData().iterator().next().asFeatureVector();

    // here's the first instance in the stub dataset. 
    // ======================================================
    //    new BasicInstance<Integer, SparseFeatureVector>(
    //        0, null, ""+(i++), new BasicSparseFeatureVector(new int[]{0, 4}, new double[]{1, 2}),
    //        newAnnotations(new long[]{0}, new int[]{0})) 
    // ======================================================
    // It has: 
    //  label=0
    //  features: 0x1; 4x2
    //  1 annotation: annotator=0 said class=0
    
    double[] expectedLogCountOfY = model.getLogCountOfY().clone();
    DoubleArrays.expToSelf(expectedLogCountOfY);
    expectedLogCountOfY[model.getY()[docIndex]] -= lambda;
    DoubleArrays.logToSelf(expectedLogCountOfY);
    double[][][] expectedCountOfJYAndA = Matrices.clone(model.getCountOfJYAndA());
    // N.B. this would usually be lambda instead of 1, but isn't implemented because we NEVER scale an instance unless it has no annotions
    expectedCountOfJYAndA[0][model.getY()[docIndex]][0] -= 1;
    double[][] expectedCountOfMAndX = Matrices.clone(model.getCountOfMAndX());
    expectedCountOfMAndX[model.getM()[docIndex]][0] -= lambda*1;
    expectedCountOfMAndX[model.getM()[docIndex]][4] -= lambda*2;
    double[] expectedNumFeaturesPerM = model.getNumFeaturesPerM().clone();
    expectedNumFeaturesPerM[model.getM()[docIndex]] -= lambda*3;
    double[][] expectedLogCountOfYAndM = Matrices.clone(model.getLogCountOfYAndM());
    Matrices.expToSelf(expectedLogCountOfYAndM);
    expectedLogCountOfYAndM[model.getY()[docIndex]][model.getM()[docIndex]] -= lambda;
    Matrices.logToSelf(expectedLogCountOfYAndM);
    double[] expectedLogSumCountOfYAndM = model.getLogSumCountOfYAndM().clone();
    DoubleArrays.expToSelf(expectedLogSumCountOfYAndM);
    expectedLogSumCountOfYAndM[model.getY()[docIndex]] -= lambda;
    DoubleArrays.logToSelf(expectedLogSumCountOfYAndM);
    
    model.decrementCounts(docIndex, instance, lambda);

    double[] actualLogCountOfY = model.getLogCountOfY();
    double[][][] actualCountOfJYAndA = model.getCountOfJYAndA();
    double[][] actualCountOfMAndX = model.getCountOfMAndX();
    double[] actualNumFeaturesPerM = model.getNumFeaturesPerM();
    double[][] actualLogCountOfYAndM = model.getLogCountOfYAndM();
    double[] actualLogSumCountOfYAndM = model.getLogSumCountOfYAndM();

    assertThat(actualLogCountOfY).isEqualTo(expectedLogCountOfY, Delta.delta(1e-10));
    for (int j=0; j<numAnnotators; j++){
      for (int c=0; c<numLabels; c++){
        assertThat(actualCountOfJYAndA[j][c]).isEqualTo(expectedCountOfJYAndA[j][c], Delta.delta(1e-10));
      }
    }
    for (int d=0; d<numLabels; d++){
      assertThat(actualCountOfMAndX[d]).isEqualTo(expectedCountOfMAndX[d], Delta.delta(1e-10));
    }
    assertThat(actualNumFeaturesPerM).isEqualTo(expectedNumFeaturesPerM, Delta.delta(1e-10));
    for (int c=0; c<numLabels; c++){
      assertThat(actualLogCountOfYAndM[c]).isEqualTo(expectedLogCountOfYAndM[c], Delta.delta(1e-10));
    }
    assertThat(actualLogSumCountOfYAndM).isEqualTo(expectedLogSumCountOfYAndM, Delta.delta(1e-10));
  }
  
  @Test
  public void testIncrementCounts(){
    for (double lambda=0; lambda<=1; lambda+=0.1){
      testIncrementCounts(lambda);
    }
  }
  private void testIncrementCounts(double lambda){

    ModelBuilder builder = newStubBuilder();
    int docIndex = 0; 
    int numLabels = builder.getData().getInfo().getNumClasses();
    int numAnnotators = builder.getPriors().getNumAnnotators();
    
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    SparseFeatureVector instance = builder.getData().iterator().next().asFeatureVector();
    int nextY = 2;
    int nextM = 1;

    // here's the first instance in the stub dataset. 
    // ======================================================
    //    new BasicInstance<Integer, SparseFeatureVector>(
    //        0, null, ""+(i++), new BasicSparseFeatureVector(new int[]{0, 4}, new double[]{1, 2}),
    //        newAnnotations(new long[]{0}, new int[]{0})) 
    // ======================================================
    // It has: 
    //  label=0
    //  features: 0x1; 4x2
    //  1 annotation: annotator=0 said class=0
    
    double[] expectedLogCountOfY = model.getLogCountOfY().clone();
    DoubleArrays.expToSelf(expectedLogCountOfY);
    expectedLogCountOfY[nextY] += lambda;
    DoubleArrays.logToSelf(expectedLogCountOfY);
    double[][][] expectedCountOfJYAndA = Matrices.clone(model.getCountOfJYAndA());
    // N.B. this would usually be lambda instead of 1, but isn't implemented because we NEVER scale an instance unless it has no annotions
    expectedCountOfJYAndA[0][nextY][0] += 1; 
    double[][] expectedCountOfMAndX = Matrices.clone(model.getCountOfMAndX());
    expectedCountOfMAndX[nextM][0] += lambda*1;
    expectedCountOfMAndX[nextM][4] += lambda*2;
    double[] expectedNumFeaturesPerM = model.getNumFeaturesPerM().clone();
    expectedNumFeaturesPerM[nextM] += lambda*3;
    double[][] expectedLogCountOfYAndM = Matrices.clone(model.getLogCountOfYAndM());
    Matrices.expToSelf(expectedLogCountOfYAndM);
    expectedLogCountOfYAndM[nextY][nextM] += lambda;
    Matrices.logToSelf(expectedLogCountOfYAndM);
    double[] expectedLogSumCountOfYAndM = model.getLogSumCountOfYAndM().clone();
    DoubleArrays.expToSelf(expectedLogSumCountOfYAndM);
    expectedLogSumCountOfYAndM[nextY] += lambda;
    DoubleArrays.logToSelf(expectedLogSumCountOfYAndM);
    
    model.incrementCounts(docIndex, instance, nextY, nextM, lambda);

    double[] actualLogCountOfY = model.getLogCountOfY();
    double[][][] actualCountOfJYAndA = model.getCountOfJYAndA();
    double[][] actualCountOfMAndX = model.getCountOfMAndX();
    double[] actualNumFeaturesPerM = model.getNumFeaturesPerM();
    double[][] actualLogCountOfYAndM = model.getLogCountOfYAndM();
    double[] actualLogSumCountOfYAndM = model.getLogSumCountOfYAndM();

    assertThat(actualLogCountOfY).isEqualTo(expectedLogCountOfY, Delta.delta(1e-10));
    for (int j=0; j<numAnnotators; j++){
      for (int c=0; c<numLabels; c++){
        assertThat(actualCountOfJYAndA[j][c]).isEqualTo(expectedCountOfJYAndA[j][c], Delta.delta(1e-10));
      }
    }
    for (int d=0; d<numLabels; d++){
      assertThat(actualCountOfMAndX[d]).isEqualTo(expectedCountOfMAndX[d], Delta.delta(1e-10));
    }
    assertThat(actualNumFeaturesPerM).isEqualTo(expectedNumFeaturesPerM, Delta.delta(1e-10));
    for (int c=0; c<numLabels; c++){
      assertThat(actualLogCountOfYAndM[c]).isEqualTo(expectedLogCountOfYAndM[c], Delta.delta(1e-10));
    }
    assertThat(actualLogSumCountOfYAndM).isEqualTo(expectedLogSumCountOfYAndM, Delta.delta(1e-10));
  }
  
  @Test
  public void testIncrementDecrementCounts(){
    ModelBuilder builder = newStubBuilder();
    for (int docIndex=0; docIndex<builder.getData().getInfo().getNumDocuments(); docIndex++){
      for (double lambda=0; lambda<=1; lambda+=0.1){
        testIncrementDecrementCounts(builder, docIndex, lambda);
      }
    }
  }
  private void testIncrementDecrementCounts(ModelBuilder builder, int docIndex, double lambda){
    int numLabels = builder.getData().getInfo().getNumClasses();
    int numAnnotators = builder.getPriors().getNumAnnotators();
    BlockCollapsedMultiAnnModel model = (BlockCollapsedMultiAnnModel)builder.build();
    
    SparseFeatureVector instance = builder.getData().iterator().next().asFeatureVector();
    
    double[] expectedLogCountOfY = model.getLogCountOfY();
    double[][][] expectedCountOfJYAndA = model.getCountOfJYAndA();
    double[][] expectedCountOfMAndX = model.getCountOfMAndX();
    double[] expectedNumFeaturesPerM = model.getNumFeaturesPerM();
    double[][] expectedLogCountOfYAndM = model.getLogCountOfYAndM();
    double[] expectedLogSumCountOfYAndM = model.getLogSumCountOfYAndM();
    
    // this operation shouldn't change anything
    int currY = model.getY()[docIndex];
    int currM = model.getM()[docIndex];
    model.decrementCounts(docIndex, instance, lambda);
    model.incrementCounts(docIndex, instance, currY, currM, lambda);
    
    double[] actualLogCountOfY = model.getLogCountOfY();
    double[][][] actualCountOfJYAndA = model.getCountOfJYAndA();
    double[][] actualCountOfMAndX = model.getCountOfMAndX();
    double[] actualNumFeaturesPerM = model.getNumFeaturesPerM();
    double[][] actualLogCountOfYAndM = model.getLogCountOfYAndM();
    double[] actualLogSumCountOfYAndM = model.getLogSumCountOfYAndM();
    
    // all counts should remain unchanged
    assertThat(actualLogCountOfY).isEqualTo(expectedLogCountOfY, Delta.delta(1e-10));
    for (int j=0; j<numAnnotators; j++){
      for (int c=0; c<numLabels; c++){
        assertThat(actualCountOfJYAndA[j][c]).isEqualTo(expectedCountOfJYAndA[j][c], Delta.delta(1e-10));
      }
    }
    for (int d=0; d<numLabels; d++){
      assertThat(actualCountOfMAndX[d]).isEqualTo(expectedCountOfMAndX[d], Delta.delta(1e-10));
    }
    assertThat(actualNumFeaturesPerM).isEqualTo(expectedNumFeaturesPerM, Delta.delta(1e-10));
    for (int c=0; c<numLabels; c++){
      assertThat(actualLogCountOfYAndM[c]).isEqualTo(expectedLogCountOfYAndM[c], Delta.delta(1e-10));
    }
    assertThat(actualLogSumCountOfYAndM).isEqualTo(expectedLogSumCountOfYAndM, Delta.delta(1e-10));
  }
  
}
