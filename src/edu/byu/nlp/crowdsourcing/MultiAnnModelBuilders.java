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
package edu.byu.nlp.crowdsourcing;

import java.util.Arrays;
import java.util.Map;

import org.apache.commons.math3.linear.SparseRealMatrix;
import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;

import edu.byu.nlp.crowdsourcing.ModelInitialization.AssignmentInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.BaselineInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.LabeledDataInitializer;
import edu.byu.nlp.crowdsourcing.ModelInitialization.MaxMarginalInitializer;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.dataset.SparseFeatureVectors;
import edu.byu.nlp.math.SparseRealMatrices;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.Iterables2;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 */
public class MultiAnnModelBuilders {

  
  public interface MultiAnnModelBuilder {
    MultiAnnModelBuilder setPriors(PriorSpecification priors);

    MultiAnnModelBuilder setDocumentWeights(double[] documentWeights);

    MultiAnnModelBuilder setData(Dataset data);

    MultiAnnModelBuilder setYInitializer(AssignmentInitializer yInitializer);

    MultiAnnModelBuilder setMInitializer(AssignmentInitializer mInitializer);

    MultiAnnModelBuilder setInitialTemp(double initialTemp);

    MultiAnnModelBuilder setRnd(RandomGenerator rnd);

    MultiAnnModel build();
  }
  
  public abstract static class AbstractMultiAnnModelBuilder implements MultiAnnModelBuilder{
    private PriorSpecification priors;
    private AssignmentInitializer yInitializer, mInitializer;
    private Dataset data;
    private int[] y;
    private int[] m;
    private double initialTemp = 0.0;
    private RandomGenerator rnd;
    private double[] documentWeights;

    public PriorSpecification getPriors() {
      return priors;
    }

    public AbstractMultiAnnModelBuilder setPriors(PriorSpecification priors) {
      this.priors = priors;
      return this;
    }

    public AbstractMultiAnnModelBuilder setDocumentWeights(double[] documentWeights) {
      this.documentWeights = documentWeights;
      return this;
    }

    public double getDocumentWeight(int docIndex){
      return documentWeights==null? 1: documentWeights[docIndex]; 
    }
    
    public Dataset getData() {
      return data;
    }

    public AbstractMultiAnnModelBuilder setData(Dataset data) {
      this.data = data;
      return this;
    }

    public int[] getY() {
      return y;
    }

    @VisibleForTesting
    public AbstractMultiAnnModelBuilder setY(int[] y) {
      this.y = y;
      return this;
    }

    public AbstractMultiAnnModelBuilder setYInitializer(AssignmentInitializer yInitializer) {
      this.yInitializer = yInitializer;
      return this;
    }

    public AbstractMultiAnnModelBuilder setMInitializer(AssignmentInitializer mInitializer) {
      this.mInitializer = mInitializer;
      return this;
    }

    public int[] getM() {
      return m;
    }

    @VisibleForTesting
    public AbstractMultiAnnModelBuilder setM(int[] m) {
      this.m = m;
      return this;
    }

    public double getInitialTemp() {
      return initialTemp;
    }

    public AbstractMultiAnnModelBuilder setInitialTemp(double initialTemp) {
      this.initialTemp = initialTemp;
      return this;
    }

    public RandomGenerator getRnd() {
      return rnd;
    }

    public AbstractMultiAnnModelBuilder setRnd(RandomGenerator rnd) {
      this.rnd = rnd;
      return this;
    }

    private double[] logCountOfY() {
      double[] counts = new double[numLabels()];
      Arrays.fill(counts, priors.getBTheta());
      for (int c=0; c<y.length; c++) {
        // pfelt: weighted documents are scaled accordingly
        double docWeight = getDocumentWeight(c);
        counts[y[c]] += docWeight;
      }
      DoubleArrays.logToSelf(counts);
      return counts;
    }

    private double[][] countOfMAndX() {
      double[][] countOfMAndX = new double[numLabels()][numFeatures()];

      for (int k = 0; k < countOfMAndX.length; k++) {
        Arrays.fill(countOfMAndX[k], priors.getBPhi());
      }

      int docIndex = 0;
      for (DatasetInstance instance : data) {
        // pfelt: weighted documents are scaled accordingly
        double docWeight = getDocumentWeight(docIndex);
        SparseFeatureVector vec = instance.asFeatureVector().copy();
        SparseFeatureVectors.multiplyToSelf(vec, docWeight);
        vec.addTo(countOfMAndX[m[docIndex]]);
        ++docIndex;
      }
      return countOfMAndX;
    }

    private double[][] countOfYAndM() {
      double[][] countOfYAndM = new double[numLabels()][numLabels()];
      CrowdsourcingUtils.initializeConfusionMatrixWithPrior(countOfYAndM, priors.getBMu(), priors.getCMu());

      for (int i = 0; i < y.length; i++) {
        // pfelt: weighted documents are scaled accordingly
        double docWeight = getDocumentWeight(i);
        countOfYAndM[y[i]][m[i]] += docWeight;
      }
      return countOfYAndM;
    }

    private double[][][] countOfJYAndA() {
      double[][][] countOfJTruthAndAnn = new double[numAnnotators()][numLabels()][numLabels()];

      for (int j = 0; j < countOfJTruthAndAnn.length; j++) {
        // TODO(rhaertel): put convenience method in Priors class.
        CrowdsourcingUtils.initializeConfusionMatrixWithPrior(countOfJTruthAndAnn[j], priors.getBGamma(j),
            priors.getCGamma());
      }

      int docIndex = 0;
      for (DatasetInstance instance : data) {
        // pfelt: weighted documents are scaled accordingly
        // note: because we only have plans to downweight docs that have no 
        // annotations, docWeight should never actually be anything but 1
        // in this case, but I include the logic here for correctness, for 
        // reference, and in case we ever decide to scale a set of annotated documents
        // in the future.
        double docWeight = getDocumentWeight(docIndex);
        
        SparseRealMatrix annotations = instance.getAnnotations().getLabelAnnotations();
        for (int annotator=0; annotator<data.getInfo().getNumAnnotators(); annotator++){
          for (int annval=0; annval<data.getInfo().getNumClasses(); annval++){
            // NOTE: this will need to update every time a new annotation comes in!!!!!
            countOfJTruthAndAnn[annotator][y[docIndex]][annval] += docWeight * annotations.getEntry(annotator, annval);
          }
        }
        ++docIndex;
      }
      return countOfJTruthAndAnn;
    }

    private double[] docSize() {
      double[] docSize = new double[numInstances()];
      int docIndex = 0;
      for (DatasetInstance instance : data) {
        // pfelt: weighted documents will be scaled in the msum math rather than 
        // mutating the data in place (to avoid side-effects). For readability, 
        // we similarly leave document lengths unchanged here and apply the scaling
        // when calculating m sums. 
        docSize[docIndex] = instance.asFeatureVector().sum();
        ++docIndex;
      }
      return docSize;
    }

    private int[][] docJCount() {
      int[][] docJCount = new int[numInstances()][numAnnotators()];
      for (Enumeration<DatasetInstance> e : Iterables2.enumerate(data)) {
        int i = e.getIndex();
        // pfelt: weighted documents are scaled accordingly
        double docWeight = getDocumentWeight(i);
        for (int ann=0; ann<data.getInfo().getNumAnnotators(); ann++){
          docJCount[i][ann] += SparseRealMatrices.rowSum(ann, e.getElement().getAnnotations().getLabelAnnotations()) * docWeight;
        }
//        SparseRealMatrices.rowSum(i, e.getElement().getAnnotations().getLabelAnnotations());
//        Set<Entry<Long, Collection<DatasetInstance>>> entries = e.getElement().getAnnotations().asMap().entrySet();
//        for (Entry<Long, Collection<DatasetInstance>> entry : entries) {
//          docJCount[i][entry.getKey().intValue()] += entry.getValue().size()*docWeight;
//        }
      }
      return docJCount;
    }

    private int numInstances() {
      return data.getInfo().getNumDocuments();
    }

    private int numLabels() {
      return data.getInfo().getNumClasses();
    }

    private int numFeatures() {
      return data.getInfo().getNumFeatures();
    }

    private int numAnnotators() {
      return priors.getNumAnnotators();
    }

    public MultiAnnModel build() {
      Preconditions.checkNotNull(data);
      Preconditions.checkNotNull(priors);
      Preconditions.checkNotNull(rnd);
      Preconditions.checkNotNull(yInitializer);
      Preconditions.checkNotNull(mInitializer);
      
      // precalculate which instance corresponds to which index. This will 
      // be used by external code when mapping y to specific predictions
      Map<String,Integer> instanceIndices = Datasets.instanceIndices(data);

      // (pfelt) initialize y and m. This was moved from ModelBuilder() into 
      // ModelBuilder.build() from  because the baseline initializer
      // does not know how to initialize weights until 
      // build time (it's based on current annotations)
      if (y==null){
        y = new int[data.getInfo().getNumDocuments()];
      }
      if (m==null){
        m = new int[data.getInfo().getNumDocuments()];
      }
      yInitializer.setData(data, instanceIndices);
      yInitializer.initialize(y);
      mInitializer.setData(data, instanceIndices);
      mInitializer.initialize(m);
      // ensure y values for labeled data are set to the correct values before doing any sampling
      LabeledDataInitializer labeledYInitializer = new LabeledDataInitializer();
      labeledYInitializer.setData(data, instanceIndices);
      labeledYInitializer.initialize(y);
      // give model implementor one last say (they may use this to rectify m/y assignments)
      hookPostInitYandM(y,m);
      
      // TODO(rhaertel): add more validation
      Preconditions.checkNotNull(y);
      Preconditions.checkNotNull(m);
      Preconditions.checkArgument(y.length == data.getInfo().getNumDocuments());
      Preconditions.checkArgument(m.length == y.length);

//      // Note (pfelt): we could pre-weight the data, which would lead to correct 
//      // behavior naturally in both model counts n^phi as well as the data 
//      // vectors used while calculating msums. However, having the model builder mutate 
//      // the data is unexpected, and could lead to badness later on. So I'm going 
//      // to do this differently, altering the n^phi initialization logic 
//      // and the msums logic, instead.
//      // (pfelt) nigam-style data balancing
//      int i=0;
//      for (DatasetInstance inst: data){
//        double docTotal = inst.asFeatureVector().sum();
//        double scalingFactor = documentWeights[i]/docTotal;
//        SparseFeatureVectors.multiplyToSelf(inst.asFeatureVector(), scalingFactor);
//        i++;
//      }

      double[][] countOfMAndX = countOfMAndX();
      double[][] countOfYAndM = countOfYAndM();
      double[] logSumCountOfYAndM = Matrices.sumOverSecond(countOfYAndM);
      DoubleArrays.logToSelf(logSumCountOfYAndM);

      // cache all gold labels for debugging 
      int[] gold = Datasets.concealedLabels(data, instanceIndices);
      
      // cache available labels for convenience (the model *may* use these)
      Dataset labeledData = Datasets.divideInstancesWithObservedLabels(data).getFirst();
      Map<Integer,Integer> instanceLabels = Maps.newHashMap();
      for (DatasetInstance inst: labeledData){
        instanceLabels.put(instanceIndices.get(inst.getInfo().getSource()), inst.getObservedLabel());
      }
      
      // Alias for better readability
      double[][] logCountOfYAndM = countOfYAndM;
      Matrices.logToSelf(logCountOfYAndM);
      double[][][] countOfJYAndA = countOfJYAndA();
      return build(priors, data, instanceIndices, instanceLabels, 
          Datasets.compileDenseAnnotations(data), 
          y, m, logCountOfY(),
          logCountOfYAndM, countOfMAndX, countOfJYAndA, logSumCountOfYAndM,
          Matrices.sumOverSecond(countOfMAndX), docSize(), numAnnsPerJAndY(countOfJYAndA),
          docJCount(), initialTemp, documentWeights, gold, rnd);
    }
    
    /**
     * This method is called after initializing Y and M but before deriving 
     * counts so that additional tweaks 
     * can be done to Y and M's state, as needed (the neutered model 
     * uses this to ensure y and m are precisely the same)
     */
    protected void hookPostInitYandM(int[] y, int[] m){}
    
    protected abstract MultiAnnModel build(PriorSpecification priors, Dataset data, Map<String,Integer> instanceIndices, 
        Map<Integer,Integer> instanceLabels, int[][][] a, int[] y, int[] m, 
        double[] logCountOfY, double[][] logCountOfYAndM, double[][] countOfMAndX,
        double[][][] countOfJYAndA, double[] logSumCountOfYAndM, double[] numFeaturesPerM,
        double[] docSize, double[][] numAnnsPerJAndY, int[][] docJCount, 
        double initialTemp, double[] documentWeights, int[] gold, RandomGenerator rnd);

  } // end class AbstractMultiAnnModelBuilder
  

  
  
  /**
   * Initialize a model builder with the same assignment initializer for Y and M
   */
  public static MultiAnnModelBuilder initModelBuilder(MultiAnnModelBuilder builder, PriorSpecification priors, Dataset data,
      AssignmentInitializer assignmentInit, RandomGenerator rnd) {
    return initModelBuilder(builder, priors, data, assignmentInit, assignmentInit, rnd);
  }

  public static MultiAnnModelBuilder initModelBuilder(MultiAnnModelBuilder builder, 
      PriorSpecification priors, Dataset data,
      AssignmentInitializer yAssignmentInit,
      AssignmentInitializer mAssignmentInit, RandomGenerator rnd) {
    builder.setPriors(priors);
    builder.setData(data);
    builder.setYInitializer(yAssignmentInit);
    builder.setMInitializer(mAssignmentInit);
    builder.setRnd(rnd);
    return builder;
  }

  public static MultiAnnModelBuilder initModelBuilderWithUniform(MultiAnnModelBuilder builder, 
      PriorSpecification priors, Dataset data, RandomGenerator rnd) {
    AssignmentInitializer initializer = ModelInitialization
        .newUniformAssignmentInitializer(data.getInfo().getNumClasses(), rnd);
    return initModelBuilder(builder, priors, data, initializer, rnd);
  }

  public static MultiAnnModelBuilder initModelBuilderWithBaselineInit(MultiAnnModelBuilder builder, 
      PriorSpecification priors, Dataset data, RandomGenerator rnd) {
    AssignmentInitializer initializer = new BaselineInitializer(rnd);
//    initModelBuilder(builder, priors, data, initializer, rnd);
    AssignmentInitializer mInitializer = new BaselineInitializer(rnd, true);
//    // note (pfelt): the below doesn't perturb m settings according to the prior 
//    AssignmentInitializer mInitializer = new ModelInitialization.NoisyInitializer(new BaselineInitializer(rnd, true), 1-priors.getBMu(), data.getInfo().getNumClasses(), rnd);
//    AssignmentInitializer mInitializer = new ModelInitialization.UniformAssignmentInitializer(data.getInfo().getNumClasses(), rnd);
    return initModelBuilder(builder, priors, data, initializer, mInitializer, rnd);
  }

  public static MultiAnnModelBuilder initModelBuilderWithSerializedChains(MultiAnnModelBuilder builder, 
      PriorSpecification priors, Dataset data, int[][] yChains, int[][] mChains,
      RandomGenerator rnd) {
    AssignmentInitializer yInitializer = new MaxMarginalInitializer(yChains,
        data.getInfo().getNumClasses());
    AssignmentInitializer mInitializer = new MaxMarginalInitializer(mChains,
        data.getInfo().getNumClasses());
    return initModelBuilder(builder, priors, data, yInitializer, mInitializer, rnd);
  }

//  // Randomly initializes a model for the given dataset. All variables are
//  // initialized uniform
//  // randomly. Only uses data in data.unlabeledData().
//  public static BlockCollapsedMultiAnnModelNeutered newModelWithUniform(
//      PriorSpecification priors, Dataset data, RandomGenerator rnd) {
//    initModelBuilderWithUniform(priors, data, rnd).build();
//  }


  public static double[][] numAnnsPerJAndY(double[][][] jya) {
    // note (pfelt): document weighting is built into jya, so we need do nothing here
    double[][] numAnnsPerJAndY = new double[jya.length][jya[0].length];
    for (int j = 0; j < numAnnsPerJAndY.length; j++) {
      numAnnsPerJAndY[j] = Matrices.sumOverSecond(jya[j]);
    }
    return numAnnsPerJAndY;
  }
  
}
